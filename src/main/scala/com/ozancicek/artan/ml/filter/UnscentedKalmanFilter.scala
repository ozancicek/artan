package com.ozancicek.artan.ml.filter

import com.ozancicek.artan.ml.linalg.LinalgUtils
import com.ozancicek.artan.ml.state.{KalmanState, KalmanUpdate}
import com.ozancicek.artan.ml.state.{StatefulTransformer}
import com.ozancicek.artan.ml.stats._
import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Matrix}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{LAPACK, BLAS}
import org.apache.spark.sql._
import scala.math.{pow}


class UnscentedKalmanFilter(
  val stateSize: Int,
  val measurementSize: Int,
  override val uid: String)
  extends KalmanTransformer[
    UnscentedKalmanStateCompute,
    UnscentedKalmanStateEstimator]
  with KalmanUpdateParams with HasStateMean with HasStateCovariance with HasFadingFactor
  with HasProcessFunction with HasMeasurementFunction {

  def this(
    measurementSize: Int,
    stateSize: Int) = {
    this(measurementSize, stateSize, Identifiable.randomUID("unscentedKalmanFilter"))
  }

  def getSigma = new MerweScaledSigmaPoints(stateSize, 0.3, 2.0, 0.1)

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

  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  def setCalculateLoglikelihood: this.type = set(calculateLoglikelihood, true)

  def setCalculateMahalanobis: this.type = set(calculateMahalanobis, true)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = withExtraColumns(filter(dataset))

  def stateUpdateFunc = new UnscentedKalmanStateEstimator(
    getStateMean,
    getStateCov,
    getFadingFactor,
    getSigma,
    getProcessFunctionOpt,
    getMeasurementFunctionOpt
  )
}


private[ml] class UnscentedKalmanStateEstimator(
  val stateMean: Vector,
  val stateCov: Matrix,
  val fadingFactor: Double,
  val sigma: SigmaPoints,
  val processFunction: Option[(Vector, Matrix) => Vector],
  val measurementFunction: Option[(Vector, Matrix) => Vector])
  extends KalmanStateUpdateFunction[UnscentedKalmanStateCompute] {

  val kalmanCompute = new UnscentedKalmanStateCompute(
    fadingFactor,
    sigma,
    processFunction,
    measurementFunction)
}


private[ml] class UnscentedKalmanStateCompute(
  fadingFactor: Double,
  sigma: SigmaPoints,
  processFunc: Option[(Vector, Matrix) => Vector],
  measurementFunc: Option[(Vector, Matrix) => Vector]) extends KalmanStateCompute {

  private[ml] def predict(
    state: KalmanState,
    process: KalmanUpdate): KalmanState = {

    val sigmaPoints = sigma.sigmaPoints(state.mean.toDense, state.covariance.toDense)

    val processModel = process.processModel.get
    val processFunction = processFunc.getOrElse(
      (in: Vector, model: Matrix) => processModel.multiply(in))
    val processNoise = process.processNoise.get

    val fadingFactorSquare = pow(fadingFactor, 2)
    val stateSigmaPoints = sigmaPoints.map { sigmas =>
      val newSigmas = processFunction(sigmas, processModel).toDense
      (process.control, process.controlFunction) match {
        case (Some(vec), Some(func)) => BLAS.gemv(1.0, func, vec, 1.0, newSigmas)
        case _ =>
      }
      newSigmas
    }

    val (stateMean, stateCov) = sigma.unscentedTransform(
      stateSigmaPoints,
      processNoise.toDense,
      fadingFactorSquare)

    KalmanState(
      state.groupKey,
      state.index + 1,
      stateMean,
      stateCov,
      state.residual,
      state.residualCovariance)
  }


  private[ml] def estimate(
    state: KalmanState,
    process: KalmanUpdate): KalmanState = {

    val (stateMean, stateCov) = (state.mean.toDense, state.covariance.toDense)
    val stateSigmaPoints = sigma.sigmaPoints(stateMean, stateCov)

    val measurementModel = process
      .measurementModel.get

    val measurementFunction = measurementFunc.getOrElse(
      (in: Vector, model: Matrix) => model.multiply(in))

    val measurementNoise = process.measurementNoise.get.toDense

    val measurementSigmaPoints = stateSigmaPoints
      .map(x => measurementFunction(x, measurementModel).toDense)

    val (estimateMean, estimateCov) = sigma.unscentedTransform(
      measurementSigmaPoints,
      measurementNoise,
      1.0)

    val fadingFactorSquare = scala.math.pow(fadingFactor, 2)

    val crossCov = DenseMatrix.zeros(state.mean.size, process.measurement.get.size)
    stateSigmaPoints.zip(measurementSigmaPoints).zipWithIndex.foreach {
      case ((stateSigma, measurementSigma), i) => {

        val stateResidual = stateSigma.copy
        BLAS.axpy(-1.0, stateMean, stateResidual)

        val measurementResidual = measurementSigma.copy
        BLAS.axpy(-1.0, estimateMean, measurementResidual)

        BLAS.dger(sigma.covWeights(i) * fadingFactorSquare,
                  stateResidual,
                  measurementResidual,
                  crossCov)
      }
    }

    val gain = crossCov.multiply(LinalgUtils.pinv(estimateCov))

    val residual = process.measurement.get.copy.toDense
    BLAS.axpy(-1.0, estimateMean, residual)

    val newMean = stateMean.copy
    BLAS.gemv(1.0, gain, residual, 1.0, newMean)

    val covUpdate = gain.multiply(estimateCov).multiply(gain.transpose)

    val newCov = DenseMatrix.zeros(stateCov.numRows, stateCov.numCols)
    BLAS.axpy(fadingFactorSquare, stateCov, newCov)
    BLAS.axpy(-1.0, covUpdate, newCov)

    KalmanState(
      state.groupKey,
      state.index,
      newMean,
      newCov,
      residual,
      estimateCov)
  }

  private[ml] def update(
    state: KalmanState,
    process: KalmanUpdate): KalmanState = {
    estimate(predict(state, process), process)
  }
}


trait SigmaPoints extends Serializable {

  val stateSize: Int

  val meanWeights: DenseVector
  val covWeights: DenseVector

  def sigmaPoints(mean: DenseVector, cov: DenseMatrix): List[DenseVector]

  def unscentedTransform(
    sigmaPoints: List[DenseVector],
    noise: DenseMatrix,
    fadeFactor: Double): (DenseVector, DenseMatrix) = {

    val newMean = new DenseVector(Array.fill(noise.numCols) {0.0})
    sigmaPoints.zipWithIndex.foreach {
      case(sigma, i) => {
        BLAS.axpy(meanWeights(i), sigma, newMean)
      }
    }

    val newCov = noise.copy
    sigmaPoints.zipWithIndex.foreach { case(sigma, i) =>
      val residual = sigma.copy
      BLAS.axpy(-1.0, newMean, residual)
      BLAS.dger(covWeights(i) * fadeFactor, residual, residual, newCov)
    }
    (newMean, newCov)
  }
}


class MerweScaledSigmaPoints(
  val stateSize: Int,
  val alpha: Double,
  val beta: Double,
  val kappa: Double) extends SigmaPoints {

  private val lambda = pow(alpha, 2) * (stateSize + kappa) - stateSize
  private val initConst = 0.5 / (stateSize + lambda)

  val meanWeights = {
    val weights = Array.fill(2 * stateSize + 1) { initConst }
    weights(0) = lambda / (stateSize + lambda)
    new DenseVector(weights)
  }

  val covWeights = {
    val weights = Array.fill(2 * stateSize + 1) { initConst }
    weights(0) = lambda / (stateSize + lambda) + (1 - pow(alpha, 2) + beta)
    new DenseVector(weights)
  }

  def sigmaPoints(mean: DenseVector, cov: DenseMatrix) = {
    val covUpdate = DenseMatrix.zeros(cov.numRows, cov.numCols)
    BLAS.axpy((lambda + stateSize), cov, covUpdate)
    val sqrt = LinalgUtils.sqrt(covUpdate)

    val (pos, neg) = sqrt.rowIter.foldLeft((List(mean), List[DenseVector]())) {
      case (coeffs, right) => {
        val meanPos = mean.copy
        BLAS.axpy(1.0, right, meanPos)
        val meanNeg = mean.copy
        BLAS.axpy(-1.0, right, meanNeg)
        (meanPos::coeffs._1, meanNeg::coeffs._2)
      }
    }
    pos.reverse:::neg.reverse
  }

}
