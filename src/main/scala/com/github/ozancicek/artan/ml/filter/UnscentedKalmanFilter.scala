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
import com.github.ozancicek.artan.ml.state.{KalmanInput, KalmanState}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS

import scala.math.pow

/**
 * Unscented Kalman Filter (UKF), implemented with a stateful spark Transformer for running parallel filters /w spark
 * dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
 * spark transformations, which can be used in both streaming and batch applications.
 *
 * Similar to Extended Kalman Filter (EKF), UKF is aimed for filtering nonlinear systems. With deterministic sampling
 * techniques, it picks up a minimal sample points and propogates them through state transition and measurement
 * functions. It doesn't need specifying jacobian functions. Instead, sampling algorithms with their hyperparameters
 * can be selected from available implementations. All linear kalman filter parameters are also valid
 * for UKF. In addition to Linear Kalman Filter parameters, following functions
 * can be specified assuming a state (x_k) with size n_s, and measurements (z_k) with size n_m;
 *
 * - f(x_k, F_k), process function for state transition. x_k is state vector and F_k is process model.
 *   Should output a vector with size (n_s)
 *
 * - h(x_k, H_k), measurement function. Should output a vector with size (n_m)
 *
 *
 * UKF will predict & estimate the state according to following equations;
 *
 * State prediction:
 *  x_k = f(x_k-1, F_k) + B_k * u_k + w_k
 *
 * Measurement incorporation:
 *  z_k = h(x_k, H_k) + v_k
 *
 * Where v_k and w_k are noise vectors drawn from zero mean, Q_k and R_k covariance
 * distributions.
 *
 * The default values of system matrices will not give you a functioning filter, but they will be initialized
 * with reasonable values given the state and measurement sizes. All of the inputs to the filter can
 * be specified with a dataframe column which will allow you to have different value across measurements/filters,
 * or you can specify a constant value across all measurements/filters.
 *
 * @param stateSize size of the state vector
 * @param measurementSize size of the measurement vector
 */
class UnscentedKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[
    UnscentedKalmanStateCompute,
    UnscentedKalmanStateSpec,
    UnscentedKalmanFilter]
  with HasProcessFunction with HasMeasurementFunction with AdaptiveNoiseParams with SigmaPointsParams {

  def this(
    stateSize: Int,
    measurementSize: Int) = {
    this(stateSize, measurementSize, Identifiable.randomUID("unscentedKalmanFilter"))
  }

  protected val defaultStateKey: String = "filter.unscentedKalmanFilter.defaultStateKey"

  /**
   * Set process function which governs state transition. It should accept the current stateVector
   * and processModel as arguments, and should output a vector of size (stateSize)
   */
  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  /**
   * Set measurement function which maps state to measurements. It should accept the current state vector
   * and measurementModel matrix as arguments, and should output a measurement vector of size (measurementSize)
   */
  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  /**
   * Set sigma point sampling algorithm for unscented transformation. Allowed values are;
   *
   * - 'merwe' (default) - E. Merwe (2000) The Unscented kalman filter for nonlinear estimation
   * - 'julier' - S. Julier (1997) A new extension to kalman filter to nonlinear systems
   */
  def setSigmaPoints(value: String): this.type = set(sigmaPoints, value)

  /**
   * Set alpha parameter for merwe algorithm
   *
   * Default is 0.3.
   */
  def setMerweAlpha(value: Double): this.type = set(merweAlpha, value)

  /**
   * Set beta parameter for merwe algorithm.
   *
   * Default is 2.0, tuned for gaussian noise.
   */
  def setMerweBeta(value: Double): this.type = set(merweBeta, value)

  /**
   * Set kappa parameter for merwe algorithm
   *
   * Default is 0.1. Suggested value is (3 - stateSize)
   */
  def setMerweKappa(value: Double): this.type = set(merweKappa, value)

  /**
   * Set kappa parameter for julier algorithm
   *
   * Default is 1.0.
   */
  def setJulierKappa(value: Double): this.type = set(julierKappa, value)

  /**
   * Enable adaptive process noise according to B. Zheng (2018) RAUKF paper
   */
  def setEnableAdaptiveProcessNoise: this.type = set(adaptiveProcessNoise, true)

  override def copy(extra: ParamMap): UnscentedKalmanFilter =  {
    val that = new UnscentedKalmanFilter(stateSize, measurementSize)
    copyValues(that, extra)
  }

  protected def stateUpdateSpec: UnscentedKalmanStateSpec = new UnscentedKalmanStateSpec(
    getFadingFactor,
    getSigmaPoints,
    getProcessFunctionOpt,
    getMeasurementFunctionOpt,
    outputResiduals,
    getAdaptiveNoiseParamSet,
    getSlidingLikelihoodWindow
  )
}

/**
 * Function spec for UKF.
 */
private[filter] class UnscentedKalmanStateSpec(
    val fadingFactor: Double,
    val sigma: SigmaPoints,
    val processFunction: Option[(Vector, Matrix) => Vector],
    val measurementFunction: Option[(Vector, Matrix) => Vector],
    val storeResidual: Boolean,
    val adaptiveNoiseParamSet: AdaptiveNoiseParamSet,
    val likelihoodWindow: Int)
  extends KalmanStateUpdateSpec[UnscentedKalmanStateCompute] {

  val kalmanCompute = new UnscentedKalmanStateCompute(
    fadingFactor,
    sigma,
    processFunction,
    measurementFunction,
    adaptiveNoiseParamSet)
}

/**
 * Class responsible for calculating UKF updates based on E. Merwe (2000) paper.
 */
private[filter] class UnscentedKalmanStateCompute(
    fadingFactor: Double,
    sigma: SigmaPoints,
    processFunc: Option[(Vector, Matrix) => Vector],
    measurementFunc: Option[(Vector, Matrix) => Vector],
    adaptiveNoiseParams: AdaptiveNoiseParamSet) extends KalmanStateCompute {

  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState = {

    val sigmaPoints = sigma.sigmaPoints(state.state.toDense, state.stateCovariance.toDense)

    val processModel = process.processModel.get
    val processFunction = processFunc.getOrElse(
      (in: Vector, model: Matrix) => processModel.multiply(in))
    val processNoise = state.processNoise.getOrElse(process.processNoise.get)

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
      state.stateIndex + 1, stateMean, stateCov,
      state.residual, state.residualCovariance,
      state.processNoise, state.slidingLoglikelihood)
  }

  def getAdaptiveProcessNoise(
    currentNoise: DenseMatrix,
    residual: DenseVector,
    estimateCov: DenseMatrix,
    gain: DenseMatrix): Option[DenseMatrix] = {
    if (adaptiveNoiseParams.isAdaptiveProcessNoise){
      val zeros = new DenseVector(Array.fill(residual.size) {0.0})
      val sqMah = LinalgUtils.squaredMahalanobis(residual, zeros, estimateCov)
      val noise = if (sqMah > adaptiveNoiseParams.adaptiveProcessNoiseThreshold) {
        val factor =  adaptiveNoiseParams.adaptiveProcessNoiseAlpha * adaptiveNoiseParams.adaptiveProcessNoiseThreshold
        val weight = scala.math.max(
          adaptiveNoiseParams.adaptiveProcessNoiseLambda,
          (sqMah - factor)/sqMah
        )
        val update = DenseMatrix.zeros(residual.size, residual.size)
        BLAS.dger(weight, residual, residual, update)

        val noise = gain.multiply(update).multiply(gain.transpose)
        BLAS.axpy(1.0-weight, currentNoise, noise)
        Some(noise)
      }
      else {
        None
      }
      noise
    }
    else {
      None
    }
  }

  def estimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean,
    likelihoodWindow: Int): KalmanState = {

    val (stateMean, stateCov) = (state.state.toDense, state.stateCovariance.toDense)

    /* Differently from E. Merwe (2000) paper, sigma points are re-calculated from predicted state rather than
    estimated state from previous time step. Produces marginally better estimates.*/
    val stateSigmaPoints = sigma.sigmaPoints(stateMean, stateCov)
    val measurementModel = process.measurementModel.get

    // Default measurement function is measurementModel * state
    val measurementFunction = measurementFunc.getOrElse(
      (in: Vector, model: Matrix) => model.multiply(in))
    val measurementNoise = process.measurementNoise.get.toDense

    // Propagate state through measurement function & perform unscented transform
    val measurementSigmaPoints = stateSigmaPoints
      .map(x => measurementFunction(x, measurementModel).toDense)
    val (estimateMean, estimateCov) = sigma
      .unscentedTransform(measurementSigmaPoints, measurementNoise, 1.0)
    val fadingFactorSquare = scala.math.pow(fadingFactor, 2)

    val crossCov = DenseMatrix.zeros(state.state.size, process.measurement.get.size)
    stateSigmaPoints.zip(measurementSigmaPoints).zipWithIndex.foreach {
      case ((stateSigma, measurementSigma), i) => {

        val stateResidual = stateSigma.copy
        BLAS.axpy(-1.0, stateMean, stateResidual)
        val measurementResidual = measurementSigma.copy
        BLAS.axpy(-1.0, estimateMean, measurementResidual)

        BLAS.dger(
          sigma.covWeights(i) * fadingFactorSquare,
          stateResidual, measurementResidual, crossCov)
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

    val (res, resCov) = if (storeResidual) (Some(residual), Some(estimateCov)) else (None, None)

    val processNoise = getAdaptiveProcessNoise(
      state.processNoise.getOrElse(process.processNoise.get).toDense,
      residual,
      estimateCov,
      gain
    )

    val ll = updateSlidingLikelihood(state.slidingLoglikelihood, likelihoodWindow, res, resCov)

    KalmanState(
      state.stateIndex, newMean, newCov, res, resCov, processNoise, ll)
  }
}

/**
 * Base trait for sigma point algorithms for performing unscented transform
 */
private[filter] trait SigmaPoints extends Serializable {

  val stateSize: Int

  val meanWeights: DenseVector

  val covWeights: DenseVector

  // Not stored as a matrix due to columnwise operations
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


private[filter] class JulierSigmaPoints(val stateSize: Int, val kappa: Double) extends SigmaPoints {

  private val initConst = 0.5/(stateSize + kappa)

  val meanWeights: DenseVector = {
    val weights = Array.fill(2 * stateSize + 1) { initConst }
    weights(0) = kappa / (kappa + stateSize)
    new DenseVector(weights)
  }

  val covWeights: DenseVector = meanWeights

  def sigmaPoints(mean: DenseVector, cov: DenseMatrix): List[DenseVector] = {
    val covUpdate = DenseMatrix.zeros(cov.numRows, cov.numCols)
    BLAS.axpy(kappa + stateSize, cov, covUpdate)
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



private[filter] class MerweSigmaPoints(
    val stateSize: Int,
    val alpha: Double,
    val beta: Double,
    val kappa: Double) extends SigmaPoints {

  private val lambda = pow(alpha, 2) * (stateSize + kappa) - stateSize

  private val initConst = 0.5 / (stateSize + lambda)

  val meanWeights: DenseVector = {
    val weights = Array.fill(2 * stateSize + 1) { initConst }
    weights(0) = lambda / (stateSize + lambda)
    new DenseVector(weights)
  }

  val covWeights: DenseVector = {
    val weights = Array.fill(2 * stateSize + 1) { initConst }
    weights(0) = lambda / (stateSize + lambda) + (1 - pow(alpha, 2) + beta)
    new DenseVector(weights)
  }

  def sigmaPoints(mean: DenseVector, cov: DenseMatrix): List[DenseVector] = {
    val covUpdate = DenseMatrix.zeros(cov.numRows, cov.numCols)
    BLAS.axpy(lambda + stateSize, cov, covUpdate)
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


private[filter] trait HasMerweAlpha extends Params {

  final val merweAlpha: DoubleParam = new DoubleParam(
    this,
    "merweAlpha",
    "Alpha parameter for merwe sigma point algorithm. Advised to be between 0 and 1"
  )

  setDefault(merweAlpha, 0.3)

  final def getMerweAlpha: Double = $(merweAlpha)
}


private[filter] trait HasMerweBeta extends Params {

  final val merweBeta: DoubleParam = new DoubleParam(
    this,
    "merweBeta",
    "Beta parameter for merwe sigma point algorithm. 2.0 is advised for gaussian noise"
  )

  setDefault(merweBeta, 2.0)

  final def getMerweBeta: Double = $(merweBeta)
}


private[filter] trait HasMerweKappa extends Params {

  final val merweKappa: DoubleParam = new DoubleParam(
    this,
    "merweKappa",
    "Kappa parameter for merwe sigma point algorithm. Advised value is (3 - stateSize)"
  )

  setDefault(merweKappa, 0.1)

  final def getMerweKappa: Double = $(merweKappa)
}


private[filter] trait HasJulierKappa extends Params {

  final val julierKappa: DoubleParam = new DoubleParam(
    this,
    "julierKappa",
    "Kappa parameter for julier sigma point algorithm."
  )

  setDefault(julierKappa, 1.0)

  final def getJulierKappa: Double = $(julierKappa)
}

private[filter] case class AdaptiveNoiseParamSet(
    isAdaptiveProcessNoise: Boolean,
    adaptiveProcessNoiseThreshold: Double,
    adaptiveProcessNoiseLambda: Double,
    adaptiveProcessNoiseAlpha: Double)


/**
 * Helper trait for creating adaptive noise param set
 */
private[filter] trait AdaptiveNoiseParams extends Params with HasAdaptiveProcessNoise
  with HasAdaptiveProcessNoiseThreshold with HasAdaptiveProcessNoiseAlpha with HasAdaptiveProcessNoiseLambda {

  protected def getAdaptiveNoiseParamSet: AdaptiveNoiseParamSet = {
    AdaptiveNoiseParamSet(
      getAdaptiveProcessNoise,
      getAdaptiveProcessNoiseThreshold,
      getAdaptiveProcessNoiseLambda,
      getAdaptiveProcessNoiseAlpha
    )
  }
}


private[filter] trait HasAdaptiveProcessNoise extends Params {

  final val adaptiveProcessNoise: BooleanParam = new BooleanParam(
    this,
    "adaptiveProcessNoise",
    "Enable adaptive process noise according to B. Zheng(2018) RAUKF paper"
  )
  setDefault(adaptiveProcessNoise, false)

  final def getAdaptiveProcessNoise: Boolean = $(adaptiveProcessNoise)
}

private[filter] trait HasAdaptiveProcessNoiseThreshold extends Params {

  final val adaptiveProcessNoiseThreshold: DoubleParam = new DoubleParam(
    this,
    "adaptiveProcessNoiseThreshold",
    "Threshold for activating adaptive process noise, measured as mahalanobis distance from residual and" +
    "its covariance."
  )

  setDefault(adaptiveProcessNoiseThreshold, 2.0)

  final def getAdaptiveProcessNoiseThreshold: Double = $(adaptiveProcessNoiseThreshold)
}


private[filter] trait HasAdaptiveProcessNoiseLambda extends Params {

  final val adaptiveProcessNoiseLambda: DoubleParam = new DoubleParam(
    this,
    "adaptiveProcessNoiseLambda",
    "Weight factor controlling the stability of noise updates. Should be between 0 and 1"
  )

  setDefault(adaptiveProcessNoiseLambda, 0.9)

  final def getAdaptiveProcessNoiseLambda: Double = $(adaptiveProcessNoiseLambda)
}

private[filter] trait HasAdaptiveProcessNoiseAlpha extends Params {

  final val adaptiveProcessNoiseAlpha: DoubleParam = new DoubleParam(
    this,
    "adaptiveProcessNoiseAlpha",
    "Weight factor controlling the senstivity of noise updates. Should be greater than 0.0" +
    "Large values give more influence to lambda factor"
  )

  setDefault(adaptiveProcessNoiseAlpha, 1.0)

  final def getAdaptiveProcessNoiseAlpha: Double = $(adaptiveProcessNoiseAlpha)
}


/**
 * Trait for parameters of sigma point algorithms.
 */
private[filter] trait SigmaPointsParams extends HasMerweAlpha with HasMerweBeta with HasMerweKappa with HasJulierKappa {

  def stateSize: Int

  final val sigmaPoints: Param[String] = new Param[String](
    this,
    "sigmaPoints",
    "sigma pints"
  )
  setDefault(sigmaPoints, "merwe")

  final def getSigmaPoints: SigmaPoints = {
    $(sigmaPoints) match {
      case "merwe" => new MerweSigmaPoints(stateSize, $(merweAlpha), $(merweBeta), $(merweKappa))
      case "julier" => new JulierSigmaPoints(stateSize, $(julierKappa))
      case _ => throw new Exception("Unsupported sigma point algorithm")
    }
  }
}