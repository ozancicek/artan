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

package com.github.ozancicek.artan.ml.mixture

import com.github.ozancicek.artan.ml.state.{MixtureStateFactory, PoissonMixtureInput, PoissonMixtureOutput, PoissonMixtureState, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.{PoissonDistribution, PoissonMixtureDistribution}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Online poisson mixture transformer, based on Cappe(2010) Online Expectation-Maximisation
 *
 * @param mixtureCount number of mixture components
 */
class PoissonMixture(
    val mixtureCount: Int,
    override val uid: String)
  extends StatefulTransformer[String, PoissonMixtureInput, PoissonMixtureState, PoissonMixtureOutput, PoissonMixture]
  with HasInitialRates with HasInitialRatesCol
  with HasPoissonMixtureModelCol with MixtureParams[PoissonMixture] {

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

  def setInitialRates(value: Array[Double]): PoissonMixture = set(initialRates, value)

  def setInitialRatesCol(value: String): PoissonMixture = set(initialRatesCol, value)

  /**
   * Transforms dataset of count to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val counts = dataset
      .withColumn("sample", col($(sampleCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))

    val mixtureInput = if (isSet(poissonMixtureModelCol)) {
      counts.withColumn("initialMixtureModel", col(getPoissonMixtureModelCol))
    } else {
      val mixtureModelFunc = udf(
        (weights: Seq[Double], rates: Seq[Double]) =>
          PoissonMixtureDistribution(weights, rates.map(r => PoissonDistribution(r))))
      val mixtureModelExpr = mixtureModelFunc(col("initialWeights"), col("initialRates"))
      counts
        .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
        .withColumn("initialRates", getUDFWithDefault(initialRates, initialRatesCol))
        .withColumn("initialMixtureModel", mixtureModelExpr)
    }

    asDataFrame(transformWithState(mixtureInput))
  }

  protected def stateUpdateSpec: PoissonMixtureUpdateSpec = new PoissonMixtureUpdateSpec(
    getUpdateHoldout, getMinibatchSize)

}

private[mixture] class PoissonMixtureUpdateSpec(val updateHoldout: Int, val minibatchSize: Int)
  extends MixtureUpdateSpec[
    Long,
    PoissonDistribution,
    PoissonMixtureDistribution,
    PoissonMixtureInput,
    PoissonMixtureState,
    PoissonMixtureOutput] {

  protected implicit def stateFactory: MixtureStateFactory[
    Long,
    PoissonDistribution,
    PoissonMixtureDistribution,
    PoissonMixtureState,
    PoissonMixtureOutput] = MixtureStateFactory.poissonSF

}


private[mixture] trait HasInitialRates extends Params {

  def mixtureCount: Int

  final val initialRates:  Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialRates",
    "initialRates")
  setDefault(initialRates, Array.tabulate(mixtureCount)(_ + 1.0))

  final def getInitialRates: Array[Double] = $(initialRates)
}


private[mixture] trait HasInitialRatesCol extends Params {

  final val initialRatesCol: Param[String] = new Param[String](
    this,
    "initialRatesCol",
    "initialRatesCol"
  )

  final def getInitialRatesCol: String = $(initialRatesCol)
}


private[mixture] trait HasPoissonMixtureModelCol extends Params {

  final val poissonMixtureModelCol: Param[String] = new Param[String](
    this,
    "poissonMixtureModelCol",
    "poissonMixtureModelCol")

  final def getPoissonMixtureModelCol: String = $(poissonMixtureModelCol)
}