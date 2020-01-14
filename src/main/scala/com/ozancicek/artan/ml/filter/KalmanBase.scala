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

import com.ozancicek.artan.ml.state.{KalmanState, KalmanInput, KalmanOutput}
import com.ozancicek.artan.ml.state.{StateUpdateFunction, StatefulTransformer}
import com.ozancicek.artan.ml.stats.{MultivariateGaussian}
import com.ozancicek.artan.ml.linalg.{LinalgUtils}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Matrix}
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{lit, col, udf}
import org.apache.spark.sql.types._


/**
 * Base trait for kalman input parameters & columns
 */
private[filter] trait KalmanUpdateParams extends HasMeasurementCol
  with HasMeasurementModelCol with HasMeasurementNoiseCol
  with HasProcessModelCol with HasProcessNoiseCol with HasControlCol
  with HasControlFunctionCol with HasProcessModel with HasMeasurementModel
  with HasProcessNoise with HasMeasurementNoise
  with HasCalculateMahalanobis with HasCalculateLoglikelihood {

  protected def getMeasurementExpr = col($(measurementCol)).cast(SQLDataTypes.VectorType)

  protected def getMeasurementModelExpr = {
    if (isSet(measurementModelCol)) {
      col($(measurementModelCol))
    } else {
      val default = $(measurementModel)
      val col = udf(()=>default)
      col()
    }
  }

  protected def getMeasurementNoiseExpr = {
    if (isSet(measurementNoiseCol)) {
      col($(measurementNoiseCol))
    } else {
      val default = $(measurementNoise)
      val col = udf(()=>default)
      col()
    }
  }

  protected def getProcessModelExpr = {
    if (isSet(processModelCol)) {
      col($(processModelCol))
    } else {
      val default = $(processModel)
      val col = udf(()=>default)
      col()
    }
  }

  protected def getProcessNoiseExpr = {
    if (isSet(processNoiseCol)) {
      col($(processNoiseCol))
    } else {
      val default = $(processNoise)
      val col = udf(()=>default)
      col()
    }
  }

  protected def getControlExpr = {
    if (isSet(controlCol)) {
      col($(controlCol))
    } else {
      lit(null).cast(SQLDataTypes.VectorType)
    }
  }

  protected def getControlFunctionExpr = {
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
      require(schema(colname).dataType == t, s"$colname must be of $t")
    }
  }

}

/**
 * Base trait for kalman filter transformers.
 *
 * @tparam Compute Type responsible for calculating the next state
 * @tparam StateUpdate Type responsible for progressing the state with a compute instance
 * @tparam ImplType Implementing class type
 */
private[filter] abstract class KalmanTransformer[
  Compute <: KalmanStateCompute,
  StateUpdate <: KalmanStateUpdateFunction[Compute],
  ImplType <: KalmanTransformer[Compute, StateUpdate, ImplType]]
  extends StatefulTransformer[String, KalmanInput, KalmanState, KalmanOutput, ImplType]
    with KalmanUpdateParams with HasInitialState with HasInitialCovariance with HasFadingFactor {

  implicit val stateKeyEncoder = Encoders.STRING

  def setInitialState(value: Vector): ImplType = set(initialState, value).asInstanceOf[ImplType]

  def setInitialCovariance(value: Matrix): ImplType = set(initialCovariance, value).asInstanceOf[ImplType]

  def setFadingFactor(value: Double): ImplType = set(fadingFactor, value).asInstanceOf[ImplType]

  def setProcessModel(value: Matrix): ImplType = set(processModel, value).asInstanceOf[ImplType]

  def setProcessNoise(value: Matrix): ImplType = set(processNoise, value).asInstanceOf[ImplType]

  def setMeasurementModel(value: Matrix): ImplType = set(measurementModel, value).asInstanceOf[ImplType]

  def setMeasurementNoise(value: Matrix): ImplType = set(measurementNoise, value).asInstanceOf[ImplType]

  def setMeasurementCol(value: String): ImplType = set(measurementCol, value).asInstanceOf[ImplType]

  def setProcessModelCol(value: String): ImplType = set(processModelCol, value).asInstanceOf[ImplType]

  def setProcessNoiseCol(value: String): ImplType = set(processNoiseCol, value).asInstanceOf[ImplType]

  def setMeasurementModelCol(value: String): ImplType = set(measurementModelCol, value).asInstanceOf[ImplType]

  def setMeasurementNoiseCol(value: String): ImplType = set(measurementNoiseCol, value).asInstanceOf[ImplType]

  def setControlCol(value: String): ImplType = set(controlCol, value).asInstanceOf[ImplType]

  def setControlFunctionCol(value: String): ImplType = set(controlFunctionCol, value).asInstanceOf[ImplType]

  def setCalculateLoglikelihood: ImplType = set(calculateLoglikelihood, true).asInstanceOf[ImplType]

  def setCalculateMahalanobis: ImplType = set(calculateMahalanobis, true).asInstanceOf[ImplType]

  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
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

  /* Calculate optional statistics*/
  protected def withExtraColumns(dataset: Dataset[KalmanOutput]): DataFrame = {
    val df = dataset.toDF
    val withLoglikelihood = if (getCalculateLoglikelihood) {
      df.withColumn("loglikelihood", loglikelihoodUDF(col("residual"), col("residualCovariance")))
    } else {df}

    val withMahalanobis = if (getCalculateMahalanobis) {
      withLoglikelihood.withColumn("mahalanobis", mahalanobisUDF(col("residual"), col("residualCovariance")))
    } else {withLoglikelihood}

    withMahalanobis
  }

  def filter(dataset: Dataset[_]): Dataset[KalmanOutput] = {
    transformSchema(dataset.schema)
    val kalmanInputDS = toKalmanInput(dataset)
    transformWithState(kalmanInputDS)
  }

  private def toKalmanInput(dataset: Dataset[_]): DataFrame = {
    /* Get the column expressions and convert to Dataset[KalmanInput]*/
    dataset
      .withColumn("measurement", getMeasurementExpr)
      .withColumn("measurementModel", getMeasurementModelExpr)
      .withColumn("measurementNoise", getMeasurementNoiseExpr)
      .withColumn("processModel", getProcessModelExpr)
      .withColumn("processNoise", getProcessNoiseExpr)
      .withColumn("control", getControlExpr)
      .withColumn("controlFunction", getControlFunctionExpr)
  }

  protected def stateUpdateFunc: StateUpdate
}


/**
 * Base trait for kalman state update function & progressing to next state.
 * @tparam Compute Type responsible for calculating the next state
 */
private[filter] trait KalmanStateUpdateFunction[+Compute <: KalmanStateCompute]
  extends StateUpdateFunction[String, KalmanInput, KalmanState, KalmanOutput] {

  /* Member responsible for calculating next state update*/
  val kalmanCompute: Compute

  /* Initial state vector*/
  def stateMean: Vector

  /* Initial covariance matrix*/
  def stateCov: Matrix

  def updateGroupState(
    key: String,
    row: KalmanInput,
    state: Option[KalmanState]): Option[KalmanState] = {

    /* If state is empty, create initial state from input parameters*/
    val currentState = state
      .getOrElse(KalmanState(
        key,
        0L,
        stateMean.toDense,
        stateCov.toDense,
        new DenseVector(Array(0.0)),
        DenseMatrix.zeros(1, 1)))

    /* Calculate next state from kalmanCompute. If there is a measurement, progress to next state with
     * predict + estimate. If the measurement is missing, progress to the next state with just predict */
    val nextState = row.measurement match {
      case Some(m) => kalmanCompute.predictAndEstimate(currentState, row)
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
    process: KalmanInput): KalmanState

  /* Function for predicting the next state*/
  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState

  /* Apply predict + estimate */
  def predictAndEstimate(
    state: KalmanState,
    process: KalmanInput): KalmanState = estimate(predict(state, process), process)
}