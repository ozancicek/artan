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

import com.github.ozancicek.artan.ml.linalg.LinalgUtils
import com.github.ozancicek.artan.ml.state.{KalmanInput, KalmanOutput, KalmanState, StateUpdateSpec, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.{MultivariateGaussian, MultivariateGaussianDistribution}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, lit, sum, udf, window, struct}
import org.apache.spark.sql.types._

import scala.collection.immutable.Queue

/**
 * Base trait for kalman input parameters & columns
 */
private[artan] trait KalmanUpdateParams[ImplType] extends HasMeasurementCol
  with HasMeasurementModelCol with HasMeasurementNoiseCol
  with HasProcessModelCol with HasProcessNoiseCol with HasControlCol
  with HasControlFunctionCol with HasProcessModel with HasMeasurementModel
  with HasProcessNoise with HasMeasurementNoise
  with HasInitialStateMean with HasInitialStateCovariance with HasFadingFactor
  with HasInitialStateMeanCol with HasInitialStateCovarianceCol with HasInitialStateDistributionCol
  with HasOutputSystemMatrices with HasMultiStepPredict {

  /**
   * Set the initial state vector with size (stateSize).
   *
   * It will be applied to all states. If the state timeouts and starts receiving
   * measurements after timeout, it will again start from this initial state vector. Default is zero. For different
   * initial state vector across filters or measurements, set the dataframe column with setInitialStateCol
   * @group setParam
   */
  def setInitialStateMean(value: Vector): ImplType = set(initialStateMean, value).asInstanceOf[ImplType]

  /**
   * Set the column corresponding to initial state vector.
   *
   * The vectors in the column should be of size (stateSize).
   * @group setParam
   */
  def setInitialStateMeanCol(value: String): ImplType = set(initialStateMeanCol, value).asInstanceOf[ImplType]

  /**
   * Set the initial covariance matrix with dimensions (stateSize, stateSize)
   *
   * It will be applied to all states. If the state timeouts and starts receiving
   * measurements after timeout, it will again start from this initial covariance vector. Default is identity matrix.
   * @group setParam
   */
  def setInitialStateCovariance(value: Matrix): ImplType = set(initialStateCovariance, value).asInstanceOf[ImplType]

  /**
   * Set the column corresponding to initial covariance matrix.
   *
   * The matrices in the column should be of dimensions (stateSize, statesize).
   * @group setParam
   */
  def setInitialStateCovarianceCol(value: String): ImplType = set(initialStateCovarianceCol, value)
    .asInstanceOf[ImplType]

  /**
   * Set the column corresponding to initial distribution from a struct column rather than individual
   * mean and covariance columns.
   * @group setParam
   */
  def setInitialStateDistributionCol(value: String): ImplType = set(initialStateDistributionCol, value)
    .asInstanceOf[ImplType]

  /**
   * Fading factor for giving more weights to more recent measurements. If needed, it should be greater than one.
   * Typically set around 1.01 ~ 1.05. Default is 1.0, which will result in equally weighted measurements.
   * @group setParam
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
   * @group setParam
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
   * @group setParam
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
   * @group setParam
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
   * @group setParam
   */
  def setMeasurementNoise(value: Matrix): ImplType = set(measurementNoise, value).asInstanceOf[ImplType]

  /**
   * Set the column corresponding to measurements.
   *
   * The vectors in the column should be of size (measurementSize). null values are allowed,
   * which will result in only state prediction step.
   * @group setParam
   */
  def setMeasurementCol(value: String): ImplType = set(measurementCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input process model matrices.
   *
   * Process model matrices should have dimensions (stateSize, stateSize)
   * @group setParam
   */
  def setProcessModelCol(value: String): ImplType = set(processModelCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input process noise matrices.
   *
   * Process noise matrices should have dimensions (stateSize, stateSize)
   * @group setParam
   */
  def setProcessNoiseCol(value: String): ImplType = set(processNoiseCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input measurement model matrices
   *
   * Measurement model matrices should have dimensions (stateSize, measurementSize)
   * @group setParam
   */
  def setMeasurementModelCol(value: String): ImplType = set(measurementModelCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input measurement noise matrices.
   *
   * Measurement noise matrices should have dimensions (measurementSize, measurementSize)
   * @group setParam
   */
  def setMeasurementNoiseCol(value: String): ImplType = set(measurementNoiseCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input control matrices.
   *
   * Control matrices should have dimensions (stateSize, controlVectorSize). null values are allowed, which will
   * result in state transition without control input
   * @group setParam
   */
  def setControlFunctionCol(value: String): ImplType = set(controlFunctionCol, value).asInstanceOf[ImplType]

  /**
   * Set the column for input control vectors.
   *
   * Control vectors should have compatible size with control function (controlVectorSize). The product of
   * control matrix & vector should produce a vector with stateSize. null values are allowed,
   * which will result in state transition without control input.
   * @group setParam
   */
  def setControlCol(value: String): ImplType = set(controlCol, value).asInstanceOf[ImplType]

  /**
   * Enable outputting system matrices
   *
   * Default is false
   * @group setParam
   */
  def setOutputSystemMatrices: ImplType = set(outputSystemMatrices, true).asInstanceOf[ImplType]

  /**
   * Number of predict steps after a single predict&update cycle
   *
   * Default is 0
   * @group setParam
   */
  def setMultiStepPredict(value: Int):ImplType = set(multiStepPredict, value).asInstanceOf[ImplType]

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
  with KalmanUpdateParams[ImplType] with HasCalculateMahalanobis with HasCalculateLoglikelihood
  with HasCalculateSlidingLikelihood with HasSlidingLikelihoodWindow with HasMultipleModelMeasurementWindowDuration
  with HasMultipleModelAdaptiveEstimationEnabled {

  protected implicit val stateKeyEncoder = Encoders.STRING

  def stateSize: Int

  def measurementSize: Int
  /**
   * Optionally calculate loglikelihood of each measurement & add it to output dataframe. Loglikelihood is calculated
   * from residual vector & residual covariance matrix.
   *
   * Not enabled by default.
   * @group setParam
   */
  def setCalculateLoglikelihood: ImplType = set(calculateLoglikelihood, true).asInstanceOf[ImplType]

  /**
   * Optionally calculate mahalanobis distance metric of each measurement & add it to output dataframe.
   * Mahalanobis distance is calculated from residual vector & residual covariance matrix.
   *
   * Not enabled by default.
   * @group setParam
   */
  def setCalculateMahalanobis: ImplType = set(calculateMahalanobis, true).asInstanceOf[ImplType]

  /**
   * Optionally calculate likelihood in a sliding window.
   *
   * Not enabled by default.
   * @group setParam
   */
  def setCalculateSlidingLikelihood: ImplType = set(calculateSlidingLikelihood, true).asInstanceOf[ImplType]


  /**
   * Set number of consecutive measurements for total likelihood calculation
   * @group setParam
   */
  def setSlidingLikelihoodWindow(value: Int): ImplType = {
    set(calculateSlidingLikelihood, true)
      .set(slidingLikelihoodWindow, value).asInstanceOf[ImplType]
  }

  /**
   * Set window duration for grouping measurements into same window for MMAE filter
   * @group setParam
   */
  def setMultipleModelMeasurementWindowDuration(value: String): ImplType = {
    set(multipleModelMeasurementWindowDuration, value)
      .asInstanceOf[ImplType]
  }

  /**
   * Optionally enable MMAE output mode
   * @group setParam
   */
  def setEnableMultipleModelAdaptiveEstimation: ImplType = {
    set(multipleModelAdaptiveEstimationEnabled, true).asInstanceOf[ImplType]
  }

  /**
   * Applies the transformation to dataset schema
   */
  def transformSchema(schema: StructType): StructType = {
    validateWatermarkColumns(schema)
    asDataFrameTransformSchema(outEncoder.schema)
  }

  private def loglikelihoodUDF = udf((residual: Vector, covariance: Matrix) => {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    MultivariateGaussian.logpdf(residual.toDense, zeroMean, covariance.toDense)
  })

  private def mahalanobisUDF = udf((residual: Vector, covariance: Matrix) => {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    LinalgUtils.mahalanobis(residual.toDense, zeroMean, covariance.toDense)
  })

  private def checkLoglikelihood(df: DataFrame): DataFrame = {
    if (getCalculateLoglikelihood) {
      df.withColumn("loglikelihood", loglikelihoodUDF(col("residual.mean"), col("residual.covariance")))
    }
    else {
      df
    }
  }

  private def checkLoglikelihoodSchema(fields: List[StructField]): List[StructField] = {
    if (getCalculateLoglikelihood) StructField("loglikelihood", DoubleType)::fields else fields
  }

  private def checkMahalanobis(df: DataFrame): DataFrame = {
    if (getCalculateMahalanobis) {
      df.withColumn("mahalanobis", mahalanobisUDF(col("residual.mean"), col("residual.covariance")))
    }
    else {
      df
    }
  }

  private def checkMahalanobisSchema(fields: List[StructField]): List[StructField] = {
    if (getCalculateMahalanobis) StructField("mahalanobis", DoubleType)::fields else fields
  }

  private def checkSlidingLikelihood(df: DataFrame): DataFrame = {
    if (!getCalculateSlidingLikelihood) df.drop("slidingLikelihood") else df
  }

  private def checkSlidingLikelihoodSchema(fields: List[StructField]): List[StructField] = {
    if (!getCalculateSlidingLikelihood) fields.filterNot(f => f.name == "slidingLikelihood") else fields
  }

  private def checkResidual(df: DataFrame): DataFrame = {
    if (outputResiduals) df else df.drop("residual")
  }

  private def checkResidualSchema(fields: List[StructField]): List[StructField] = {
    if (outputResiduals) fields else fields.filterNot(f => f.name =="residual")
  }

  private def checkSystemMatrices(df: DataFrame): DataFrame = {
    if (getOutputSystemMatrices) df else df.drop("processModel", "processNoise", "measurementModel")
  }

  private def checkSystemMatricesSchema(fields: List[StructField]): List[StructField] = {
    if (getOutputSystemMatrices) fields else fields
      .filterNot(f => Set("processModel", "processNoise", "measurementModel").contains(f.name))
  }

  // Output residuals if any of mahalanobis or likelihood calculations are requested
  protected def outputResiduals: Boolean = {
    getCalculateLoglikelihood || getCalculateMahalanobis || getCalculateSlidingLikelihood
  }

  override protected def asDataFrame(in: Dataset[KalmanOutput]): DataFrame = {
    val outDF = super.asDataFrame(in)

    val checks = (checkLoglikelihood _)
      .andThen(checkMahalanobis _)
      .andThen(checkSlidingLikelihood _)
      .andThen(checkResidual _)
      .andThen(checkSystemMatrices _)

    checks(outDF)
  }

  override protected def asDataFrameTransformSchema(schema: StructType): StructType = {
    val outSchema = super.asDataFrameTransformSchema(schema)

    val checks = (checkLoglikelihoodSchema _)
      .andThen(checkMahalanobisSchema _)
      .andThen(checkSlidingLikelihoodSchema _)
      .andThen(checkResidualSchema _)
      .andThen(checkSystemMatricesSchema _)

    StructType(checks(outSchema.toList))
  }

  private[artan] def filter(dataset: Dataset[_]): Dataset[KalmanOutput] = {
    transformSchema(dataset.schema)
    val inDF = toKalmanInput(dataset)
    transformWithState(inDF)
  }

  private def scalMatrix = udf(
    (alpha: Double, mat: Matrix) => {
      val result = DenseMatrix.zeros(mat.numRows, mat.numCols)
      BLAS.axpy(alpha, mat.toDense, result)
      result
    }
  )

  private def scalVector = udf(
    (alpha: Double, vec: Vector) => {
      val result = vec.copy
      BLAS.scal(alpha, result)
      result
    }
  )

  /**
   * Transforms dataset of measurements to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    val stateEstimates = asDataFrame(filter(dataset))

    // If MMAE is enabled, aggregate estimated states by their sliding likelihood weights
    if (getMultipleModelAdaptiveEstimationEnabled) {
      require(getCalculateSlidingLikelihood)

      val normExpr = lit(1.0)/sum("slidingLikelihood")
      val aggVecFunc = LinalgUtils.axpyVectorAggregate(stateSize)
      val aggMatFunc = LinalgUtils.axpyMatrixAggregate(stateSize, stateSize)

      val aggStateExpr = scalVector(normExpr, aggVecFunc(col("slidingLikelihood"), col("state.mean")))
      val aggCovExpr = scalMatrix(normExpr, aggMatFunc(col("slidingLikelihood"), col("state.covariance")))

      val windowKeys = if (isSet(multipleModelMeasurementWindowDuration)) {
        Seq(window(col(getEventTimeCol), getMultipleModelMeasurementWindow).alias(getEventTimeCol))
      } else {
        Seq.empty[Column]
      }
      val groupKeys = windowKeys :+ col("stateIndex")

      stateEstimates
        .groupBy(groupKeys: _*)
        .agg(aggStateExpr.alias("mean"), aggCovExpr.alias("covariance"))
        .withColumn("state", struct("mean", "covariance"))
        .drop("mean", "covariance")
    } else {
      stateEstimates
    }
  }

  private def toKalmanInput(dataset: Dataset[_]): DataFrame = {
    /* Get the column expressions and convert to Dataset[KalmanInput]*/
    val initialStateExpr = if (isSet(initialStateDistributionCol)) {
      col(getInitialStateDistributionCol)
    }
    else {
      struct("mean", "covariance")
    }

    dataset
      .withColumn("mean", getUDFWithDefault(initialStateMean, initialStateMeanCol))
      .withColumn("covariance", getUDFWithDefault(initialStateCovariance, initialStateCovarianceCol))
      .withColumn("initialState", initialStateExpr)
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
 * @tparam Compute KalmanStateCompute implementation calculating the next state
 */
private[filter] trait KalmanStateUpdateSpec[+Compute <: KalmanStateCompute]
  extends StateUpdateSpec[String, KalmanInput, KalmanState, KalmanOutput] {

  val kalmanCompute: Compute

  /* Whether to store residual in the state */
  def storeResidual: Boolean

  def likelihoodWindow: Int

  def multiStepPredict: Int

  def getOutputProcessModel(row: KalmanInput, state: KalmanState): Option[Matrix] = row.processModel

  def getOutputProcessNoise(row: KalmanInput, state: KalmanState): Option[Matrix] = row.processNoise

  def getOutputMeasurementModel(row: KalmanInput, state: KalmanState): Option[Matrix] = row.measurementModel

  protected def stateToOutput(
    key: String,
    row: KalmanInput,
    state: KalmanState): List[KalmanOutput] = {

    val ll = if (state.slidingLoglikelihood.isEmpty) None else Some(scala.math.exp(state.slidingLoglikelihood.sum))

    val asOutput = (in: KalmanState, stepIndex: Int) => {
      KalmanOutput(
        key,
        in.stateIndex,
        stepIndex,
        in.state,
        in.residual,
        row.eventTime,
        getOutputProcessModel(row, in),
        getOutputProcessNoise(row, in),
        getOutputMeasurementModel(row, in),
        ll)
    }
    (0 until multiStepPredict).foldLeft(List(asOutput(state, 0)), state) { case ((outputs, currentState), i) =>
      val nextState = kalmanCompute.predict(currentState, row)
      val nextOutput = asOutput(nextState, i + 1)
      (nextOutput::outputs, nextState)
    }._1.reverse
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
        None,
        None,
        Queue.empty[Double]))

    /* Calculate next state from kalmanCompute. If there is a measurement, progress to next state with
     * predict + estimate. If the measurement is missing, progress to the next state with just predict */
    val nextState = row.measurement match {
      case Some(m) => kalmanCompute.predictAndEstimate(currentState, row, storeResidual, likelihoodWindow)
      case None => kalmanCompute.predict(currentState, row)
    }
    Some(nextState)
  }
}


/**
 * Base trait for kalman state computation
 */
private[filter] trait KalmanStateCompute extends Serializable {

  def updateSlidingLikelihood(
    slidingLoglikelihood: Queue[Double],
    likelihoodWindow: Int,
    residualDist: Option[MultivariateGaussianDistribution]): Queue[Double] = {

    residualDist match {
      case Some(dist) => {
        val res = dist.mean
        val cov = dist.covariance
        val zeroMean = new DenseVector(Array.fill(res.size) {0.0})
        val ll = MultivariateGaussian.logpdf(res.toDense, zeroMean, cov.toDense)
        if (slidingLoglikelihood.size >= likelihoodWindow) {
          slidingLoglikelihood.dequeue._2.enqueue(ll)
        }
        else {
          slidingLoglikelihood.enqueue(ll)
        }
      }
      case _ => slidingLoglikelihood
    }
  }

  /* Function for incorporating new measurement*/
  def estimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean,
    likelihoodWindow: Int): KalmanState

  /* Function for predicting the next state*/
  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState

  /* Apply predict + estimate */
  def predictAndEstimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean,
    likelihoodWindow: Int): KalmanState = estimate(predict(state, process), process, storeResidual, likelihoodWindow)
}