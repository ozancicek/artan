package com.ozancicek.artan.ml.filter

import com.ozancicek.artan.ml.state.{KalmanState, KalmanUpdate}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{BLAS}
import org.apache.spark.sql._


class ExtendedKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[
    ExtendedKalmanStateCompute,
    ExtendedKalmanStateEstimator]
  with KalmanUpdateParams with HasStateMean with HasStateCovariance with HasFadingFactor
  with HasProcessFunction with HasProcessStateJacobian with HasProcessNoiseJacobian
  with HasMeasurementFunction with HasMeasurementStateJacobian with HasMeasurementNoiseJacobian {

  def this(
    measurementSize: Int,
    stateSize: Int) = {
    this(measurementSize, stateSize, Identifiable.randomUID("extendedKalmanFilter"))
  }

  def setStateMean(value: Vector): this.type = set(stateMean, value)

  def setStateCovariance(value: Matrix): this.type = set(stateCov, value)

  def setFadingFactor(value: Double): this.type = set(fadingFactor, value)

  def setProcessModel(value: Matrix): this.type = set(processModel, value)

  def setProcessNoise(value: Matrix): this.type = set(processNoise, value)

  def setMeasurementModel(value: Matrix): this.type = set(measurementModel, value)

  def setMeasurementNoise(value: Matrix): this.type = set(measurementNoise, value)

  def setGroupKeyCol(value: String): this.type = set(groupKeyCol, value)

  def setMeasurementCol(value: String): this.type = set(measurementCol, value)

  def setProcessModelCol(value: String): this.type = set(processModelCol, value)

  def setProcessNoiseCol(value: String): this.type = set(processNoiseCol, value)

  def setMeasurementModelCol(value: String): this.type = set(measurementModelCol, value)

  def setMeasurementNoiseCol(value: String): this.type = set(measurementNoiseCol, value)

  def setControlCol(value: String): this.type = set(controlCol, value)

  def setControlFunctionCol(value: String): this.type = set(controlFunctionCol, value)

  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  def setProcessStateJacobian(value: (Vector, Matrix) => Matrix): this.type = set(processStateJacobian, value)

  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  def setMeasurementStateJacobian(value: (Vector, Matrix) => Matrix): this.type = set(measurementStateJacobian, value)

  def setProcessNoiseJacobian(value: (Vector, Matrix) => Matrix): this.type = set(processNoiseJacobian, value)

  def setMeasurementNoiseJacobian(value: (Vector, Matrix) => Matrix): this.type = set(measurementNoiseJacobian, value)

  def setCalculateLoglikelihood: this.type = set(calculateLoglikelihood, true)

  def setCalculateMahalanobis: this.type = set(calculateMahalanobis, true)

  override def copy(extra: ParamMap): ExtendedKalmanFilter = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = withExtraColumns(filter(dataset))

  def stateUpdateFunc = new ExtendedKalmanStateEstimator(
    getStateMean,
    getStateCov,
    getFadingFactor,
    getProcessFunctionOpt,
    getProcessStateJacobianOpt,
    getProcessNoiseJacobianOpt,
    getMeasurementFunctionOpt,
    getMeasurementStateJacobianOpt,
    getMeasurementNoiseJacobianOpt
  )
}


private[ml] class ExtendedKalmanStateEstimator(
    val stateMean: Vector,
    val stateCov: Matrix,
    val fadingFactor: Double,
    val processFunction: Option[(Vector, Matrix) => Vector],
    val processStateJacobian: Option[(Vector, Matrix) => Matrix],
    val processNoiseJacobian: Option[(Vector, Matrix) => Matrix],
    val measurementFunction: Option[(Vector, Matrix) => Vector],
    val measurementStateJacobian: Option[(Vector, Matrix) => Matrix],
    val measurementNoiseJacobian: Option[(Vector, Matrix) => Matrix])
  extends KalmanStateUpdateFunction[ExtendedKalmanStateCompute] {

  val kalmanCompute = new ExtendedKalmanStateCompute(
    fadingFactor,
    processFunction,
    processStateJacobian,
    processNoiseJacobian,
    measurementFunction,
    measurementStateJacobian,
    measurementNoiseJacobian)
}

private[ml] class ExtendedKalmanStateCompute(
    fadingFactor: Double,
    processFunc: Option[(Vector, Matrix) => Vector],
    processStateJac: Option[(Vector, Matrix) => Matrix],
    processNoiseJac: Option[(Vector, Matrix) => Matrix],
    measurementFunc: Option[(Vector, Matrix) => Vector],
    measurementStateJac: Option[(Vector, Matrix) => Matrix],
    measurementNoiseJac: Option[(Vector, Matrix) => Matrix])
  extends LinearKalmanStateCompute(fadingFactor) {

  override def progressStateMean(
    stateMean: DenseVector,
    processModel: DenseMatrix): DenseVector = {
    processFunc
      .map(f => f(stateMean, processModel).toDense)
      .getOrElse(processModel.multiply(stateMean))
  }

  override def calculateResidual(
    stateMean: DenseVector,
    measurement: DenseVector,
    measurementModel: DenseMatrix): DenseVector = {
    val residual = measurement.copy
    val newMean = measurementFunc
      .map(f => f(stateMean, measurementModel).toDense)
      .getOrElse(measurementModel.multiply(stateMean))
    BLAS.axpy(-1.0, newMean, residual)
    residual
  }

  override def getProcessModel(
    stateMean: DenseVector,
    processModel: DenseMatrix): DenseMatrix = {
    processStateJac
      .map(f => f(stateMean, processModel).toDense)
      .getOrElse(processModel)
  }

  override def getMeasurementModel(
    stateMean: DenseVector,
    measurementModel: DenseMatrix): DenseMatrix = {
    measurementStateJac
      .map(f => f(stateMean, measurementModel).toDense)
      .getOrElse(measurementModel)
  }

  override def getProcessNoise(
    stateMean: DenseVector,
    processNoise: DenseMatrix): DenseMatrix = {
    processNoiseJac
      .map { f =>
        val noiseJac = f(stateMean, processNoise).toDense
        noiseJac.multiply(processNoise).multiply(noiseJac.transpose)
      }.getOrElse(processNoise)
  }

  override def getMeasurementNoise(
    stateMean: DenseVector,
    measurementNoise: DenseMatrix): DenseMatrix = {
    measurementNoiseJac
      .map { f =>
        val noiseJac = f(stateMean, measurementNoise).toDense
        noiseJac.multiply(measurementNoise).multiply(noiseJac.transpose)
      }.getOrElse(measurementNoise)
  }

}
