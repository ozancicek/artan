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

import com.github.ozancicek.artan.ml.state.{PoissonMixtureInput, PoissonMixtureOutput, PoissonMixtureState}
import com.github.ozancicek.artan.ml.stats.{PoissonDistribution, PoissonMixtureDistribution}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Online poisson mixture estimation with stateful transformer, based on Cappe(2010) Online Expectation-Maximisation.
 *
 * Outputs an estimate for each input sample in a single pass, by replacing the E-step in EM with a stochastic
 * E-step.
 *
 * @param mixtureCount number of mixture components
 */
class PoissonMixture(val mixtureCount: Int, override val uid: String)
  extends FiniteMixture[
    Long,
    PoissonDistribution,
    PoissonMixtureDistribution,
    PoissonMixtureInput,
    PoissonMixtureState,
    PoissonMixtureOutput,
    PoissonMixture]
    with HasInitialRates with HasInitialRatesCol {

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
    if (!isSet(initialMixtureModelCol)) {
      require(
        isSet(initialRates) | isSet(initialRatesCol), "Initial rates or its dataframe column must be set")
      if (isSet(initialRatesCol)) {
        require(
          schema(getInitialRatesCol).dataType == ArrayType(DoubleType),
          "Initial probabilities column should be an array of doubles with size mixtureCount")
      }
    }
    asDataFrameTransformSchema(outEncoder.schema)
  }

  /**
   * Sets the initial poisson rates of the mixtures. The length of the array should be equal to [[mixtureCount]]
   *
   * @group setParam
   */
  def setInitialRates(value: Array[Double]): PoissonMixture = set(initialRates, value)

  /**
   * Sets the initial rates from dataframe column. Overrides the parameter set from [[setInitialRates]]
   *
   * @group setParam
   */
  def setInitialRatesCol(value: String): PoissonMixture = set(initialRatesCol, value)

  protected def buildInitialMixtureModel(dataFrame: DataFrame): DataFrame = {
    val mixtureModelFunc = udf(
      (weights: Seq[Double], rates: Seq[Double]) =>
        PoissonMixtureDistribution(weights, rates.map(r => PoissonDistribution(r))))
    val mixtureModelExpr = mixtureModelFunc(col("initialWeights"), col("initialRates"))
    dataFrame
      .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
      .withColumn("initialRates", getUDFWithDefault(initialRates, initialRatesCol))
      .withColumn("initialMixtureModel", mixtureModelExpr)
  }
}


private[mixture] trait HasInitialRates extends Params {

  def mixtureCount: Int

  /**
   * Initial poisson rates of the mixtures.
   *
   * @group param
   */
  final val initialRates:  Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialRates",
    "Initial poisson rates of the mixtures. The length of the array should be equal to [[mixtureCount]]")
  setDefault(initialRates, Array.tabulate(mixtureCount)(_ + 1.0))

  /**
   * Getter for initial poisson rates parameter
   *
   * @group getParam
   */
  final def getInitialRates: Array[Double] = $(initialRates)
}


private[mixture] trait HasInitialRatesCol extends Params {

  /**
   * Initial rates from dataframe column
   *
   * @group param
   */
  final val initialRatesCol: Param[String] = new Param[String](
    this,
    "initialRatesCol",
    "Initial poisson rates from dataframe column. Overrides the [[initialRates]] parameter"
  )

  /**
   * Getter for initial rates
   *
   * @group getParam
   */
  final def getInitialRatesCol: String = $(initialRatesCol)
}