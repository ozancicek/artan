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

import com.github.ozancicek.artan.ml.state.{RLSInput, RLSOutput, RLSState, StateUpdateSpec, StatefulTransformer}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._


/**
 * Recursive formulation of least squares with exponential weighting & regularization, implemented with
 * a stateful spark Transformer for running parallel filters /w spark dataframes. Transforms an input dataframe
 * of observations to a dataframe of model parameters using stateful spark transformations, which can be used
 * in both streaming and batch applications.
 *
 * Let w denote the model parameters and w_est denote our prior belief. RLS minimizes following regularization
 * & weighted SSE terms;
 *
 * (w - w_est).T * (lambda^(-N-1) * P)^(-1) * (w - w_est) + Sum(lambda(N - j)*(d_k - u_k.T * w),  k = 0,1, .. N)
 *
 * Where:
 *  - lambda: forgetting factor, or exponential weighting factor. Between 0 and 1.
 *  - P: regularization matrix. Smaller values increseas the weight of regularization term, whereas larger values
 *    increases the weight of weighted SSE term.
 *  - d_k, u_k: label and features vector at time step k.
 *
 * @param featuresSize Size of the features vector
 */
class RecursiveLeastSquaresFilter(
    val featuresSize: Int,
    override val uid: String)
  extends StatefulTransformer[String, RLSInput, RLSState, RLSOutput, RecursiveLeastSquaresFilter]
  with HasLabelCol with HasFeaturesCol with HasForgettingFactor
  with HasInitialStateMean with HasRegularizationMatrix with HasInitialStateMeanCol with HasRegularizationMatrixCol {

  protected implicit val stateKeyEncoder = Encoders.STRING

  /**
   * Define state size equal to features vector size
   */
  def stateSize: Int = featuresSize

  def this(featuresSize: Int) = this(featuresSize, Identifiable.randomUID("recursiveLeastSquaresFilter"))

  protected val defaultStateKey: String = "filter.recursiveLeastSquaresFilter.defaultStateKey"

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): RecursiveLeastSquaresFilter =  {
    val that = new RecursiveLeastSquaresFilter(featuresSize)
    copyValues(that, extra)
  }

  /**
   * Set label column. Default is "label"
   * @group setParam
   */
  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol, "label")

  /**
   * Set features column. Default is "features"
   * @group setParam
   */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  setDefault(featuresCol, "features")

  /**
   * Set forgetting factor, which exponentially weights measurements to have more influence from recent measurements.
   *
   * Default value of 1.0 weights all measurements equally. With smaller values, recent measurements will have
   * more weights. Generally set around 0.95 ~ 0.99
   * @group setParam
   */
  def setForgettingFactor(value: Double): this.type = set(forgettingFactor, value)

  /**
   * Set regularization matrix governing the influence of the initial estimate (prior). Larger values will
   * remove regularization effect, making the filter behave like OLS. Use setRegularizationMatrixCol for different
   * values accross filters.
   *
   * Default is 10E5 * I
   * @group setParam
   */
  def setRegularizationMatrix(value: Matrix): this.type = set(regularizationMatrix, value)

  /**
   * Set regularization matrix factor, which results in setting the regularization matrix as
   * factor * identity matrix.
   * @group setParam
   */
  def setRegularizationMatrixFactor(value: Double): this.type = set(regularizationMatrix, getFactoredIdentity(value))
  setDefault(regularizationMatrix, getFactoredIdentity(10E5))

  /**
   * Set initial estimate for model parameters for all filters. Use setInitialEstimateCol for different
   * initial estimates across filters. Default is zero vector.
   * @group setParam
   */
  def setInitialEstimate(value: Vector): this.type = set(initialStateMean, value)

  /**
   * Set initial estimate column.
   * @group setParam
   */
  def setInitialEstimateCol(value: String): this.type = set(initialStateMeanCol, value)

  /**
   * Set regularization matrix column.
   * @group setParam
   */
  def setRegularizationMatrixCol(value: String): this.type = set(regularizationMatrixCol, value)

  private def getFactoredIdentity(value: Double): DenseMatrix = {
    new DenseMatrix(stateSize, stateSize, DenseMatrix.eye(stateSize).values.map(_ * value))
  }

  private def validateSchema(schema: StructType): Unit = {
    validateWatermarkColumns(schema)
    if (isSet(stateKeyCol)) {
      require(schema($(stateKeyCol)).dataType == StringType, "Group key column must be StringType")
    }
    require(schema($(labelCol)).dataType == DoubleType)
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType)
  }

  /**
   * Applies the transformation to dataset schema
   */
  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    asDataFrameTransformSchema(outEncoder.schema)
  }

  private[artan] def filter(dataset: Dataset[_]): Dataset[RLSOutput] = {
    transformSchema(dataset.schema)

    val rlsUpdateDS = dataset
      .withColumn("label", col($(labelCol)))
      .withColumn("features", col($(featuresCol)))
      .withColumn("initialState", getUDFWithDefault(initialStateMean, initialStateMeanCol))
      .withColumn("initialCovariance", getUDFWithDefault(regularizationMatrix, regularizationMatrixCol))

    transformWithState(rlsUpdateDS)
  }

  /**
   * Transforms dataset of features & label to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = asDataFrame(filter(dataset))

  protected def stateUpdateSpec: RecursiveLeastSquaresUpdateSpec = new RecursiveLeastSquaresUpdateSpec(
    getForgettingFactor)
}

/**
 * Function spec for calculating RLS updates.
 */
private[filter] class RecursiveLeastSquaresUpdateSpec(
    val forgettingFactor: Double)
  extends StateUpdateSpec[String, RLSInput, RLSState, RLSOutput] {

  protected def stateToOutput(key: String, row: RLSInput, state: RLSState): List[RLSOutput] = {
    List(RLSOutput(
      key,
      state.stateIndex,
      state.state,
      state.covariance,
      row.eventTime))
  }

  def updateGroupState(
    key: String,
    row: RLSInput,
    state: Option[RLSState]): Option[RLSState] = {

    val features = row.features
    val label = row.label

    val currentState = state
      .getOrElse(RLSState(0L, row.initialState, row.initialCovariance))

    val model = currentState.covariance.transpose.multiply(features)

    val gain = currentState.covariance.multiply(features)
    BLAS.scal(1.0/(forgettingFactor + BLAS.dot(model, features)), gain)
    val residual = label -  BLAS.dot(features, currentState.state)

    val estMean = currentState.state.copy
    BLAS.axpy(residual, gain, estMean)
    val gainUpdate = DenseMatrix.zeros(gain.size, features.size)
    BLAS.dger(1.0, gain, features.toDense, gainUpdate)

    val currentCov = currentState.covariance.toDense
    val covUpdate = currentCov.copy
    BLAS.gemm(-1.0, gainUpdate, currentCov, 1.0, covUpdate)

    val estCov = DenseMatrix.zeros(covUpdate.numRows, covUpdate.numCols)
    BLAS.axpy(1.0/forgettingFactor, covUpdate, estCov)

    val newState = RLSState(currentState.stateIndex + 1L, estMean, estCov)
    Some(newState)
  }
}


private[filter] trait HasForgettingFactor extends Params {

  /**
   * Forgetting factor for having more weight in recent measurements, between 0.0 and 1.0
   * Default value of 1.0 weights all measurements equally. Smaller values increases the weight in recent
   * measurements. Typically around 0.95~0.99
   * @group param
   */
  final val forgettingFactor: DoubleParam = new DoubleParam(
    this,
    "forgettingFactor",
    "Forgetting factor for having more weight in recent measurements, between 0.0 and 1.0" +
    "Default value of 1.0 weights all measurements equally. Smaller values increases the weight in recent" +
    "measurements. Typically around 0.95~0.99",
    ParamValidators.ltEq(1.0))

  setDefault(forgettingFactor, 1.0)

  /**
   * Getter for forgetting factor
   * @group getParam
   */
  final def getForgettingFactor: Double = $(forgettingFactor)
}

private[filter] trait HasRegularizationMatrix extends Params {

  def featuresSize: Int

  /**
   * Positive definite regularization matrix for RLS filter, typically a factor multiplied by identity matrix.
   * Small factors (factor>1) give more weight to the initial state, whereas large factors (>>1) decrease
   * regularization and cause RLS filter to behave like ordinary least squares.
   *
   * @group param
   */
  final val regularizationMatrix: Param[Matrix] = new Param[Matrix](
    this,
    "regularizationMatrix",
    "Positive definite regularization matrix for RLS filter, typically a factor multiplied by identity matrix." +
    "Small factors (factor>1) give more weight to the initial state, whereas large factors (>>1) decrease" +
    "regularization and cause RLS filter to behave like ordinary least squares",
    (in: Matrix) => (in.numRows == featuresSize) & (in.numCols == featuresSize))

  setDefault(regularizationMatrix, DenseMatrix.eye(featuresSize))

  /**
   * Getter for regularization matrix param
   * @group getParam
   */
  final def getRegularizationMatrix: Matrix = $(regularizationMatrix)

}

private[filter] trait HasRegularizationMatrixCol extends Params {

  def featuresSize: Int

  /**
   * Param for regularization matrix column for specifying different reg matrices across filters.
   * Overrides [[regularizationMatrix]] setting.
   * @group param
   */
  final val regularizationMatrixCol: Param[String] = new Param[String](
    this,
    "regularizationMatrixCol",
    "Regularization matrix column for specifying different reg matrices across filters")

  /**
   * Getter for regularization matrix column
   * @group getParam
   */
  final def getRegularizationMatrixCol: String = $(regularizationMatrixCol)

}
