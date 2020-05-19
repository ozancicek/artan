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

import com.github.ozancicek.artan.ml.state.{BernoulliMixtureInput, BernoulliMixtureOutput, BernoulliMixtureState}
import com.github.ozancicek.artan.ml.stats.{BernoulliDistribution, BernoulliMixtureDistribution}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Online bernoulli mixture estimator with a stateful transformer, based on Cappe (2011) Online
 * Expectation-Maximisation paper.
 *
 * Outputs an estimate for each input sample in a single pass, by replacing the E-step in EM with a stochastic
 * E-step.
 *
 * @param mixtureCount number of mixture components
 */
class BernoulliMixture(
    val mixtureCount: Int,
    override val uid: String)
  extends FiniteMixture[
    Boolean,
    BernoulliDistribution,
    BernoulliMixtureDistribution,
    BernoulliMixtureInput,
    BernoulliMixtureState,
    BernoulliMixtureOutput,
    BernoulliMixture]
  with HasInitialProbabilities with HasInitialProbabilitiesCol {

  def this(mixtureCount: Int) = this(mixtureCount, Identifiable.randomUID("BernoulliMixture"))

  protected val defaultStateKey: String = "em.BernoulliMixture.defaultStateKey"

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): BernoulliMixture =  {
    val that = new BernoulliMixture(mixtureCount)
    copyValues(that, extra)
  }

  /**
   * Applies the transformation to dataset schemas
   */
  def transformSchema(schema: StructType): StructType = {
    if (!isSet(initialMixtureModelCol)) {
      require(
        isSet(initialProbabilities) | isSet(initialProbabilitiesCol),
        "Initial probabilities or its dataframe column must be set")
      if (isSet(initialProbabilitiesCol)) {
        require(
          schema(getInitialProbabilitiesCol).dataType == ArrayType(DoubleType),
          "Initial probabilities column should be an array of doubles with size mixtureCount")
      }
    }
    asDataFrameTransformSchema(outEncoder.schema)
  }

  /**
   * Sets the initial bernoulli probabilities of the mixtures. The length of the array should be equal to mixture
   * count, each element in the array should be between 0 and 1.
   *
   * Default is equally spaced probabilities between 0 and 1
   *
   * @group setParam
   */
  def setInitialProbabilities(value: Array[Double]): BernoulliMixture = set(initialProbabilities, value)

  /**
   * Sets the initial probabilities from dataframe column to set different probabilities across different models.
   * Overrides the parameter set by [[setInitialProbabilities]]
   *
   * @group setParam
   */
  def setInitialProbabilitiesCol(value: String): BernoulliMixture = set(initialProbabilitiesCol, value)

  protected def buildInitialMixtureModel(dataFrame: DataFrame): DataFrame = {
    val mixtureModelFunc = udf(
      (weights: Seq[Double], probabilities: Seq[Double]) =>
        BernoulliMixtureDistribution(weights, probabilities.map(BernoulliDistribution(_))))
    val mixtureModelExpr = mixtureModelFunc(col("initialWeights"), col("initialProbabilities"))
    dataFrame
      .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
      .withColumn("initialProbabilities", getUDFWithDefault(initialProbabilities, initialProbabilitiesCol))
      .withColumn("initialMixtureModel", mixtureModelExpr)
  }
}


private[mixture] trait HasInitialProbabilities extends Params {

  def mixtureCount: Int

  private def getDefault = {
    val start = 1.0/(mixtureCount + 2)
    val interval = 1.0/(mixtureCount + 1)

    (start until (1.0 - interval) by interval).toArray
  }

  /**
   * Initial probabilities of the mixtures.
   *
   * @group param
   */
  final val initialProbabilities: Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialProbabilities",
    "Initial bernoulli probabilities of the mixtures. The length of the array should be equal to mixture" +
      "count, each element in the array should be between 0 and 1. Default is equally spaced probabilities between" +
      "0 and 1")

  setDefault(initialProbabilities, getDefault)

  /**
   * Getter for initial probabilities column
   *
   * @group getParam
   */
  final def getInitialProbabilities: Array[Double] = $(initialProbabilities)
}


private[mixture] trait HasInitialProbabilitiesCol extends Params {

  /**
   * Initial probabilities from dataframe column.
   *
   * @group param
   */
  final val initialProbabilitiesCol: Param[String] = new Param[String](
    this,
    "initialProbabilitiesCol",
    "Initial probabilities from dataframe column. Overrides the [[initialProbabilities]] parameter"
  )

  /**
   * Getter for initial probabilities column
   *
   * @group getParam
   */
  final def getInitialProbabilitiesCol: String = $(initialProbabilitiesCol)
}