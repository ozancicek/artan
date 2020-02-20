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

package com.ozancicek.artan.ml.filter

import com.ozancicek.artan.ml.state.{KalmanInput, KalmanOutput, KalmanState, StateUpdateSpec, StatefulTransformer}
import com.ozancicek.artan.ml.stats.MultivariateGaussian
import com.ozancicek.artan.ml.linalg.LinalgUtils
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{DenseVector, Matrix, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types._


import scala.reflect.runtime.universe.TypeTag

/**
 * Base trait for kalman input parameters & columns
 */
private[artan] trait KalmanUpdateParams[ImplType] extends HasMeasurementCol
  with HasMeasurementModelCol with HasMeasurementNoiseCol
  with HasProcessModelCol with HasProcessNoiseCol with HasControlCol
  with HasControlFunctionCol with HasProcessModel with HasMeasurementModel
  with HasProcessNoise with HasMeasurementNoise
  with HasInitialState with HasInitialCovariance with HasFadingFactor
  with HasInitialStateCol with HasInitialCovarianceCol {

  /**
   * Set the initial state vector with size (stateSize).
   *
   * It will be applied to all states. If the state timeouts and starts receiving
   * measurements after timeout, it will again start from this initial state vector. Default is zero. For different
   * initial state vector across filters or measurements, set the dataframe column with setInitialStateCol
   */
  def setInitialState(value: Vector): ImplType = set(initialState, value).asInstanceOf[ImplType]

  /**
   * Set the column corresponding to initial state vector.
   *
   * The vectors in the column should be of size (stateSize).
   */
  def setInitialStateCol(value: String): ImplType = set(initialStateCol, value).asInstanceOf[ImplType]

  /**
   * Set the initial covariance matrix with dimensions (stateSize, stateSize)
   *
   * It will be applied to all states. If the state timeouts and starts receiving
   * measurements after timeout, it will again start from this initial covariance vector. Default is identity matrix.
   */
  def setInitialCovariance(value: Matrix): ImplType = set(initialCovariance, value).asInstanceOf[ImplType]

  /**
   * Set the column corresponding to initial covariance matrix.
   *
   * The matrices in the column should be of dimensions (stateSize, statesize).
   */
  def setInitialCovarianceCol(value: String): ImplType = set(initialCovarianceCol, value).asInstanceOf[ImplType]

  /**
   * Fading factor for giving more weights to more recent measurements. If needed, it should be greater than one.
   * Typically set around 1.01 ~ 1.05. Default is 1.0, which will result in equally weighted measurements.
   */
  def setFadingFactor(value: Double): ImplType = set(fadingFactor, value).asInstanceOf[ImplType]

  /**
   * Set default value for process model matrix with dimensions (stateSize, stateSize) which governs state transition.
   *
   * Note that if this parameter is set through here, it will result in same process model for all filters &
   * measurements. For different process models across filters or measurements, set a dataframe column for process
   * model from setProcessModelCol.
   *
   * Default is identity matrix.
   */
  def setProcessModel(value: Matrix): ImplType = set(processModel, value).asInstanceOf[ImplType]

  /**
   * Set default value for process noise matrix with dimensions (stateSize, stateSize).
   *
   * Note that if this parameter is set through here, it will result in same process noise for all filters &
   * measurements. For different process noise values across filters or measurements, set a dataframe column
   * for process noise from setProcessNoiseCol.
   *
   * Default is identity matrix.
   */
  def setProcessNoise(value: Matrix): ImplType = set(processNoise, value).asInstanceOf[ImplType]

  /**
   * Set default value for measurement model matrix with dimensions (stateSize, measurementSize)
   * which maps states to measurement.
   *
   * Note that if this parameter is set through here, it will result in same measurement model for all filters &
   * measurements. For different measurement models across filters or measurements, set a dataframe column for
   * measurement model from setMeasurementModelCol.
   *
   * Default value maps the first state value to measurements.
   */
  def setMeasurementModel(value: Matrix): ImplType = set(measurementModel, value).asInstanceOf[ImplType]

  /**
   * Set default value for measurement noise matrix with dimensions (measurementSize, measurementSize).
   *
   * Note that if this parameter is set through here, it will result in same measurement noise for all filters &
   * measurements. For different measurement noise values across filters or measurements,
   * set a dataframe column for measurement noise from setMeasurementNoiseCol.
   *
   * Default is identity matrix.
   */
  def setMeasurementNoise(value: Matrix): ImplType = set(measurementNoise, value).asInstanceOf[ImplType]

  /**
   * Set the column corresponding to measurements.
   *
   * The vectors in the column should be of size (measurementSize). null values are allowed,
   * which will result in only state prediction step.
   */
  def setMeasurementCol(value: String): ImplType = set(measurementCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input process model matrices.
   *
   * Process model matrices should have dimensions (stateSize, stateSize)
   */
  def setProcessModelCol(value: String): ImplType = set(processModelCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input process noise matrices.
   *
   * Process noise matrices should have dimensions (stateSize, stateSize)
   */
  def setProcessNoiseCol(value: String): ImplType = set(processNoiseCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input measurement model matrices
   *
   * Measurement model matrices should have dimensions (stateSize, measurementSize)
   */
  def setMeasurementModelCol(value: String): ImplType = set(measurementModelCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input measurement noise matrices.
   *
   * Measurement noise matrices should have dimensions (measurementSize, measurementSize)
   */
  def setMeasurementNoiseCol(value: String): ImplType = set(measurementNoiseCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input control matrices.
   *
   * Control matrices should have dimensions (stateSize, controlVectorSize). null values are allowed, which will
   * result in state transition without control input
   */
  def setControlFunctionCol(value: String): ImplType = set(controlFunctionCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input control vectors.
   *
   * Control vectors should have compatible size with control function (controlVectorSize). The product of
   * control matrix & vector should produce a vector with stateSize. null values are allowed,
   * which will result in state transition without control input.
   */
  def setControlCol(value: String): ImplType = set(controlCol, value).asInstanceOf[ImplType]

  protected def getUDFWithDefault[
    DefaultType: TypeTag](defaultParam: Param[DefaultType], colParam: Param[String]): Column = {

    if (isSet(colParam)) {
      col($(colParam))
    } else {
      val defaultVal = $(defaultParam)
      val col = udf(() => defaultVal)
      col()
    }

  }

  protected def getMeasurementExpr: Column = col($(measurementCol)).cast(SQLDataTypes.VectorType)

  protected def getControlExpr: Column = {
    if (isSet(controlCol)) {
      col($(controlCol))
    } else {
      lit(null).cast(SQLDataTypes.VectorType)
    }
  }

  protected def getControlFunctionExpr: Column = {
    if (isSet(controlFunctionCol)) {
      col($(controlFunctionCol))
    } else {
      lit(null).cast(SQLDataTypes.MatrixType)
    }
  }

  protected def validateSchema(schema: StructType): Unit = {
    if (isSet(measurementModelCol)) {
      require(
        schema($(measurementModelCol)).dataType == SQLDataTypes.MatrixType,
        "Measurement model column must be MatrixType")
    }

    val vectorCols = Seq(measurementCol, controlCol)
    val matrixCols = Seq(
      measurementModelCol, measurementNoiseCol, processModelCol,
      processNoiseCol, controlFunctionCol)

    vectorCols.foreach(col=>validateColParamType(schema, col, SQLDataTypes.VectorType))
    matrixCols.foreach(col=>validateColParamType(schema, col, SQLDataTypes.MatrixType))
  }

  private def validateColParamType(schema: StructType, col: Param[String], t: DataType): Unit = {
    if (isSet(col)) {
      val colname = $(col)
      val colType = schema(colname).dataType
      require(colType == t, s"$colname must be of $t, found $colType")
    }
  }

}

/**
 * Base trait for kalman filter transformers.
 *
 * @tparam Compute Type responsible for calculating the next state
 * @tparam SpecType Type responsible for progressing the state with a compute instance
 * @tparam ImplType Implementing class type
 */
private[filter] abstract class KalmanTransformer[
  Compute <: KalmanStateCompute,
  SpecType <: KalmanStateUpdateSpec[Compute],
  ImplType <: KalmanTransformer[Compute, SpecType, ImplType]]
  extends StatefulTransformer[String, KalmanInput, KalmanState, KalmanOutput, ImplType]
    with KalmanUpdateParams[ImplType] with HasCalculateMahalanobis with HasCalculateLoglikelihood {

  implicit val stateKeyEncoder = Encoders.STRING

  /**
   * Optionally calculate loglikelihood of each measurement & add it to output dataframe. Loglikelihood is calculated
   * from residual vector & residual covariance matrix.
   *
   * Not enabled by default.
   */
  def setCalculateLoglikelihood: ImplType = set(calculateLoglikelihood, true).asInstanceOf[ImplType]

  /**
   * Optinally calculate mahalanobis distance metric of each measuremenet & add it to output dataframe.
   * Mahalanobis distance is calculated from residual vector & residual covariance matrix.
   *
   * Not enabled by default.
   */
  def setCalculateMahalanobis: ImplType = set(calculateMahalanobis, true).asInstanceOf[ImplType]

  def transformSchema(schema: StructType): StructType = {
    validateWatermarkColumns(schema)
    outEncoder.schema
  }

  private def loglikelihoodUDF = udf((residual: Vector, covariance: Matrix) => {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    MultivariateGaussian.logpdf(residual.toDense, zeroMean, covariance.toDense)
  })

  private def mahalanobisUDF = udf((residual: Vector, covariance: Matrix) => {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    LinalgUtils.mahalanobis(residual.toDense, zeroMean, covariance.toDense)
  })

  private def withLoglikelihood(df: DataFrame): DataFrame = df
    .withColumn("loglikelihood", loglikelihoodUDF(col("residual"), col("residualCovariance")))

  private def withMahalanobis(df: DataFrame): DataFrame = df
    .withColumn("mahalanobis", mahalanobisUDF(col("residual"), col("residualCovariance")))

  protected def outputResiduals: Boolean = getCalculateLoglikelihood || getCalculateMahalanobis

  private[filter] def filter(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    val inDF = toKalmanInput(dataset)
    val outDF = transformWithState(inDF).toDF
    if (outputResiduals) {
      val dfWithMahalanobis = if (getCalculateMahalanobis) withMahalanobis(outDF) else outDF
      val dfWithLoglikelihood = if (getCalculateLoglikelihood) withLoglikelihood(dfWithMahalanobis) else dfWithMahalanobis
      dfWithLoglikelihood
    }
    else {
      outDF
    }
  }

  private def toKalmanInput(dataset: Dataset[_]): DataFrame = {
    /* Get the column expressions and convert to Dataset[KalmanInput]*/
    dataset
      .withColumn("initialState", getUDFWithDefault(initialState, initialStateCol))
      .withColumn("initialCovariance", getUDFWithDefault(initialCovariance, initialCovarianceCol))
      .withColumn("measurement", getMeasurementExpr)
      .withColumn("measurementModel", getUDFWithDefault(measurementModel, measurementModelCol))
      .withColumn("measurementNoise", getUDFWithDefault(measurementNoise, measurementNoiseCol))
      .withColumn("processModel", getUDFWithDefault(processModel, processModelCol))
      .withColumn("processNoise", getUDFWithDefault(processNoise, processNoiseCol))
      .withColumn("control", getControlExpr)
      .withColumn("controlFunction", getControlFunctionExpr)
  }

  protected def stateUpdateSpec: SpecType
}


/**
 * Base trait for kalman state update spec to progress to next state.
 * @tparam Compute KalmanStateCompute implementation which calculates the next state
 */
private[filter] trait KalmanStateUpdateSpec[+Compute <: KalmanStateCompute]
  extends StateUpdateSpec[String, KalmanInput, KalmanState, KalmanOutput] {

  val kalmanCompute: Compute

  /* Whether to store residual in the state */
  def storeResidual: Boolean

  def getOutputProcessModel(row: KalmanInput, state: KalmanState): Option[Matrix] = row.processModel

  def getOutputProcessNoise(row: KalmanInput, state: KalmanState): Option[Matrix] = row.processNoise

  def getOutputMeasurementModel(row: KalmanInput, state: KalmanState): Option[Matrix] = row.measurementModel

  protected def stateToOutput(
    key: String,
    row: KalmanInput,
    state: KalmanState): List[KalmanOutput] = {

    List(
      KalmanOutput(
      key,
      state.stateIndex,
      state.state,
      state.stateCovariance,
      state.residual,
      state.residualCovariance,
      row.eventTime,
      getOutputProcessModel(row, state),
      getOutputProcessNoise(row, state),
      getOutputMeasurementModel(row, state)))
  }

  def updateGroupState(
    key: String,
    row: KalmanInput,
    state: Option[KalmanState]): Option[KalmanState] = {

    /* If state is empty, create initial state from input parameters*/
    val currentState = state
      .getOrElse(KalmanState(
        0L,
        row.initialState,
        row.initialCovariance,
        None,
        None,
        None))

    /* Calculate next state from kalmanCompute. If there is a measurement, progress to next state with
     * predict + estimate. If the measurement is missing, progress to the next state with just predict */
    val nextState = row.measurement match {
      case Some(m) => kalmanCompute.predictAndEstimate(currentState, row, storeResidual)
      case None => kalmanCompute.predict(currentState, row)
    }
    Some(nextState)
  }
}


/**
 * Base trait for kalman state computation
 */
private[filter] trait KalmanStateCompute extends Serializable {

  /* Function for incorporating new measurement*/
  def estimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean): KalmanState

  /* Function for predicting the next state*/
  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState

  /* Apply predict + estimate */
  def predictAndEstimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean): KalmanState = estimate(predict(state, process), process, storeResidual)
}