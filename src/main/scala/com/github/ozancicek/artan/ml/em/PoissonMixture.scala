/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.ozancicek.artan.ml.em

import com.github.ozancicek.artan.ml.state.{PoissonMixtureInput, PoissonMixtureOutput}
import com.github.ozancicek.artan.ml.state.{PoissonMixtureModel, PoissonMixtureState}
import com.github.ozancicek.artan.ml.state.{StateUpdateSpec, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.PoissonDistribution
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Experimental online poisson mixture transformer, based on Cappe(2010) Online Expectation-Maximisation
 *
 * @param mixtureCount number of mixture components
 */
class PoissonMixture(
    val mixtureCount: Int,
    override val uid: String)
  extends StatefulTransformer[String, PoissonMixtureInput, PoissonMixtureState, PoissonMixtureOutput, PoissonMixture]
  with HasInitialWeights with HasInitialWeightsCol with HasInitialRates with HasInitialRatesCol
  with HasPoissonMixtureModelCol with HasStepSize with HasStepSizeCol with HasCountCol {

  protected implicit val stateKeyEncoder = Encoders.STRING

  def this(mixtureCount: Int) = this(mixtureCount, Identifiable.randomUID("PoissonMixture"))

  protected val defaultStateKey: String = "em.PoissonMixture.defaultStateKey"

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): PoissonMixture =  {
    val that = new PoissonMixture(mixtureCount)
    copyValues(that, extra)
  }

  /**
   * Applies the transformation to dataset schema
   */
  def transformSchema(schema: StructType): StructType = {
    asDataFrameTransformSchema(outEncoder.schema)
  }

  def setInitialRates(value: Array[Double]): PoissonMixture = {
    set(initialRates, value)
  }

  def setInitialWeights(value: Array[Double]): PoissonMixture = {
    set(initialWeights, value)
  }

  def setInitialWeightsCol(value: String): PoissonMixture = {
    set(initialWeightsCol, value)
  }

  def setStepSize(value: Double): PoissonMixture = {
    set(stepSize, value)
  }

  def setStepSizeCol(value: String): PoissonMixture = {
    set(stepSizeCol, value)
  }
  /**
   * Transforms dataset of count to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val counts = dataset
      .withColumn("count", col($(countCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))

    val mixtureInput = if (isSet(poissonMixtureModelCol)) {
      counts.withColumn("initialMixtureModel", col(getPoissonMixtureModelCol))
    } else {
      val mixtureModelFunc = udf(
        (weights: Seq[Double], rates: Seq[Double]) =>
          PoissonMixtureModel(weights.toArray, rates.toArray.map(r => PoissonDistribution(r))))
      val mixtureModelExpr = mixtureModelFunc(col("initialWeights"), col("initialRates"))
      counts
        .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
        .withColumn("initialRates", getUDFWithDefault(initialRates, initialRatesCol))
        .withColumn("initialMixtureModel", mixtureModelExpr)
    }

    asDataFrame(transformWithState(mixtureInput))
  }

  protected def stateUpdateSpec: PoissonMixtureUpdateSpec = new PoissonMixtureUpdateSpec(5)

}

private[em] class PoissonMixtureUpdateSpec(updateHoldout: Int)
  extends StateUpdateSpec[String, PoissonMixtureInput, PoissonMixtureState, PoissonMixtureOutput] {

  protected def stateToOutput(
    key: String,
    row: PoissonMixtureInput,
    state: PoissonMixtureState): List[PoissonMixtureOutput] = {
    List(PoissonMixtureOutput(
      key,
      state.stateIndex,
      state.mixtureModel,
      row.eventTime))
  }

  def updateGroupState(
    key: String,
    row: PoissonMixtureInput,
    state: Option[PoissonMixtureState]): Option[PoissonMixtureState] = {

    val getInitialState = () => {
      val rates = row.initialMixtureModel.distributions.map(_.rate)
      val initialRateState = BLAS
        .elemMult(1.0, new DenseVector(row.initialMixtureModel.weights), new DenseVector(rates))
      val model = row.initialMixtureModel
      PoissonMixtureState(0L, row.initialMixtureModel.weights, initialRateState.values, model)
    }

    val currentState = state
      .getOrElse(getInitialState())

    val likelihood = currentState.mixtureModel.distributions.zip(currentState.mixtureModel.weights).map {
      case (dist, weight) => dist.pmf(row.count) * weight
    }

    val sumLikelihood = likelihood.sum
    val normLikelihood = likelihood.map(_ * row.stepSize/sumLikelihood)

    val weightsSummary = currentState.weightsSummary
      .zip(normLikelihood).map(s => (1 - row.stepSize) * s._1 + s._2)

    val ratesSummary = currentState.ratesSummary
      .zip(normLikelihood).map(s => (1 - row.stepSize) * s._1 + s._2 * row.count.toDouble)

    val newModel = if (currentState.stateIndex < updateHoldout) {
      currentState.mixtureModel
    } else {
      val weights = weightsSummary
      val rates = ratesSummary.zip(weights).map(s => PoissonDistribution(s._1 / s._2))
      PoissonMixtureModel(weights, rates)
    }
    val nextState = PoissonMixtureState(currentState.stateIndex + 1, weightsSummary, ratesSummary, newModel)
    Some(nextState)
  }
}


private[em] trait HasInitialRates extends Params {

  def mixtureCount: Int

  final val initialRates:  Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialRates",
    "initialRates")
  setDefault(initialRates, Array.tabulate(mixtureCount)(_ + 1.0))

  final def getInitialRates: Array[Double] = $(initialRates)
}


private[em] trait HasInitialRatesCol extends Params {

  final val initialRatesCol: Param[String] = new Param[String](
    this,
    "initialRatesCol",
    "initialRatesCol"
  )

  final def getInitialRatesCol: String = $(initialRatesCol)
}


private[em] trait HasCountCol extends Params {

  final val countCol: Param[String] = new Param[String](
    this,
    "countCol",
    "countCol")

  setDefault(countCol, "count")

  final def getCountCol: String = $(countCol)
}


private[em] trait HasPoissonMixtureModelCol extends Params {

  final val poissonMixtureModelCol: Param[String] = new Param[String](
    this,
    "poissonMixtureModelCol",
    "poissonMixtureModelCol")

  final def getPoissonMixtureModelCol: String = $(poissonMixtureModelCol)
}