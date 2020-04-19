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

import com.github.ozancicek.artan.ml.state.{PoissonMixtureInput, PoissonMixtureOutput, PoissonMixtureState}
import com.github.ozancicek.artan.ml.state.{StateUpdateSpec, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.Poisson
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
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
  with HasMixtureCoefficients with HasMixtureCoefficientsCol with HasPoissonRates with HasPoissonRatesCol
  with HasStepSize with HasStepSizeCol with HasCountCol {

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

  def setInitialPoissonRates(value: Vector): PoissonMixture = {
    set(poissonRates, value)
  }
  /**
   * Transforms dataset of count to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    val mixtureInput = dataset
      .withColumn("count", col($(countCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))
      .withColumn("initialMixtureCoefficients", getUDFWithDefault(mixtureCoefficients, mixtureCoefficientsCol))
      .withColumn("initialRates", getUDFWithDefault(poissonRates, poissonRatesCol))
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
      state.mixtureCoefficients,
      state.rates,
      row.eventTime))
  }

  def updateGroupState(
    key: String,
    row: PoissonMixtureInput,
    state: Option[PoissonMixtureState]): Option[PoissonMixtureState] = {

    val getInitialState = () => {
      val initialRateState = BLAS.elemMult(1.0, row.initialMixtureCoefficients.toDense, row.initialRates.toDense)
      PoissonMixtureState(0L, row.initialMixtureCoefficients, initialRateState, row.initialMixtureCoefficients, row.initialRates)
    }

    val currentState = state
      .getOrElse(getInitialState())

    val probs = new DenseVector(currentState.rates.toDense.values.map(r => Poisson.pmf(row.count, r)))
    val weightedProbs = BLAS.elemMult(1.0, probs, currentState.mixtureCoefficients.toDense)
    BLAS.scal(row.stepSize/weightedProbs.values.sum, weightedProbs)

    val mixtureSummary = weightedProbs.copy
    BLAS.axpy(1.0 - row.stepSize, currentState.mixtureSummary, mixtureSummary)

    val ratesSummary = weightedProbs.copy
    BLAS.scal(row.count.toDouble, ratesSummary)
    BLAS.axpy(1.0 - row.stepSize, currentState.ratesSummary, ratesSummary)

    val (coeffs, rates) = if (currentState.stateIndex < updateHoldout) {
      (currentState.mixtureCoefficients, currentState.rates)
    } else {
      val weights = mixtureSummary
      val invWeights = new DenseVector(weights.values.map(d => 1.0/d))
      val rates = BLAS.elemMult(1.0, ratesSummary, invWeights)
      (weights, rates)
    }
    val nextState = PoissonMixtureState(
      currentState.stateIndex + 1,
      mixtureSummary,
      ratesSummary,
      coeffs,
      rates
    )
    Some(nextState)
  }
}


private[em] trait HasMixtureCoefficients extends Params {

  def mixtureCount: Int

  final val mixtureCoefficients: Param[Vector] = new Param[Vector](
    this,
    "mixtureCoefficients",
    "mixtureCoefficients")

  setDefault(mixtureCoefficients, new DenseVector(Array.fill(mixtureCount) {1.0/mixtureCount}))

  final def getMixtureCoefficients: Vector = $(mixtureCoefficients)
}


private[em] trait HasMixtureCoefficientsCol extends Params {

  final val mixtureCoefficientsCol: Param[String] = new Param[String](
    this,
    "mixtureCoefficientsCol",
    "mixtureCoefficientsCol"
  )

  final def getMixtureCoefficientsCol: String = $(mixtureCoefficientsCol)
}


private[em] trait HasPoissonRates extends Params {

  def mixtureCount: Int

  final val poissonRates: Param[Vector] = new Param[Vector](
    this,
    "poissonRates",
    "poissonRates")


  final def getPoissonRates: Vector = $(poissonRates)
}


private[em] trait HasPoissonRatesCol extends Params {

  final val poissonRatesCol: Param[String] = new Param[String](
    this,
    "poissonRatesCol",
    "poissonRatesCol"
  )

  final def getPoissonRatesCol: String = $(poissonRatesCol)
}


private[em] trait HasStepSize extends Params {

  final val stepSize: Param[Double] = new DoubleParam(
    this,
    "stepSize",
    "stepSize",
    ParamValidators.lt(1.0)
  )

  setDefault(stepSize, 0.01)

  final def getStepSize: Double = $(stepSize)
}


private[em] trait HasStepSizeCol extends Params {

  final val stepSizeCol: Param[String] = new Param[String](
    this,
    "stepSizeCol",
    "stepSizeCol")

  final def getStepSizeCol: String = $(stepSizeCol)
}


private[em] trait HasCountCol extends Params {

  final val countCol: Param[String] = new Param[String](
    this,
    "countCol",
    "countCol")

  setDefault(countCol, "count")

  final def getCountCol: String = $(countCol)
}