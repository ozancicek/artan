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

package com.github.ozancicek.artan.ml.filter

import com.github.ozancicek.artan.ml.state.{LMSInput, LMSOutput, LMSState, StateUpdateSpec, StatefulTransformer}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._


/**
 * Normalized Least Mean Squares filter, implemented with a stateful spark Transformer for running parallel
 * filters /w spark dataframes. Transforms an input dataframe of observations to a dataframe of model parameters
 * using stateful spark transformations, which can be used in both streaming and batch applications.
 *
 * Belonging to stochastic gradient descent type of methods, LMS minimizes SSE on each measurement
 * based on the expectation of steepest descending gradient.
 *
 * Let w denote the model parameter vector, u denote the features vector, and d for label corresponding to u.
 * Normalized LMS computes w at step k recursively by;
 *
 * e = d - u.T * w_k-1
 * w_k = w_k-1 + m * e * u /(c + u.T*u)
 *
 * Where
 *  m: Learning rate
 *  c: Regularization constant
 *
 * @param featuresSize Size of the features vector
 */
class LeastMeanSquaresFilter(
    val featuresSize: Int,
    override val uid: String)
  extends StatefulTransformer[String, LMSInput, LMSState, LMSOutput, LeastMeanSquaresFilter]
  with HasLabelCol with HasFeaturesCol with HasInitialState with HasLearningRate with HasRegularizationConstant
  with HasInitialStateCol {

  implicit val stateKeyEncoder = Encoders.STRING

  def stateSize: Int = featuresSize

  def this(stateSize: Int) = this(stateSize, Identifiable.randomUID("leastMeanSquaresFilter"))

  protected val defaultStateKey: String = "filter.leastMeanSquaresFilter.defaultStateKey"

  override def copy(extra: ParamMap): LeastMeanSquaresFilter =  {
    val that = new LeastMeanSquaresFilter(featuresSize)
    copyValues(that, extra)
  }

  /**
   * Set label column. Default is "features"
   */
  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol, "label")

  /**
   * Set features column. Default is "features"
   */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  setDefault(featuresCol, "features")

  /**
   * Set initial estimate for all states. Must have same size with features vector. To set different initial
   * estimates, use setInitialEstimateCol
   *
   * Default is zero vector
   */
  def setInitialEstimate(value: Vector): this.type = set(initialState, value)

  /**
   * Set initial estimate vector column. It will override setInitialEstimate.
   */
  def setInitialEstimateCol(value: String): this.type = set(initialStateCol, value)

  /**
   * Set learning rate controlling the speed of convergence. Without noise, 1.0 is optimal since filter is normalized.
   *
   * Default is 1.0
   */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /**
   * Set constant for regularization, controlling the stability. Larger values increase stability but
   * degrade convergence performance. Generally set to a small constant.
   *
   * Default is 1.0
   */
  def setRegularization(value: Double): this.type = set(regularizationConstant, value)

  private def validateSchema(schema: StructType): Unit = {
    validateWatermarkColumns(schema)
    if (isSet(stateKeyCol)) {
      require(schema($(stateKeyCol)).dataType == StringType, "Group key column must be StringType")
    }
    require(schema($(labelCol)).dataType == DoubleType)
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType)
  }

  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    asDataFrameTransformSchema(outEncoder.schema)
  }

  private[artan] def filter(dataset: Dataset[_]): Dataset[LMSOutput] = {
    transformSchema(dataset.schema)
    val initialStateExpr = if (isSet(initialStateCol)) {
      col(getInitialStateCol)
    } else {
      val f = udf(() => getInitialState)
      f()
    }
    val lmsUpdateDS = dataset
      .withColumn("label", col($(labelCol)))
      .withColumn("features", col($(featuresCol)))
      .withColumn("initialState", initialStateExpr)
    transformWithState(lmsUpdateDS)
  }

  def transform(dataset: Dataset[_]): DataFrame = asDataFrame(filter(dataset))

  protected def stateUpdateSpec: LeastMeanSquaresUpdateSpec = new LeastMeanSquaresUpdateSpec(
    getLearningRate, getRegularizationConstant)

}

/**
 * Function spec for calculating Normalized LMS updates
 */
private[filter] class LeastMeanSquaresUpdateSpec(
    val learningRate: Double,
    val regularizationConstant: Double)
  extends StateUpdateSpec[String, LMSInput, LMSState, LMSOutput] {

  protected def stateToOutput(key: String, row: LMSInput, state: LMSState): List[LMSOutput] = {
    List(LMSOutput(
      key,
      state.stateIndex,
      state.state,
      row.eventTime))
  }

  def updateGroupState(
    key: String,
    row: LMSInput,
    state: Option[LMSState]): Option[LMSState] = {

    val currentState = state
      .getOrElse(LMSState(0L, row.initialState))

    val features = row.features

    val gain = features.copy
    BLAS.scal(learningRate/(regularizationConstant + BLAS.dot(features, features)), gain)

    val residual = row.label -  BLAS.dot(features, currentState.state)

    val estMean = currentState.state.copy
    BLAS.axpy(residual, gain, estMean)
    val newState = LMSState(currentState.stateIndex + 1, estMean)
    Some(newState)
  }
}


private[filter] trait HasLearningRate extends Params {

  final val learningRate: DoubleParam = new DoubleParam(
    this,
    "learningRate",
    "Learning rate for Normalized LMS. If there is no interference, the default value of 1.0 is optimal.")

  setDefault(learningRate, 1.0)

  final def getLearningRate: Double = $(learningRate)
}

private[filter] trait HasRegularizationConstant extends Params {

  final val regularizationConstant: DoubleParam = new DoubleParam(
    this,
    "regularizationConstant",
    "Regularization term for stability")

  setDefault(regularizationConstant, 1.0)

  final def getRegularizationConstant: Double = $(regularizationConstant)
}