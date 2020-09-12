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

import com.github.ozancicek.artan.ml.linalg.{LinalgOptions, LinalgUtils}
import com.github.ozancicek.artan.ml.state.{KalmanInput, KalmanState}
import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.ml.util.{DefaultParamsWritable, DefaultParamsReadable}

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
 * Measurement update:
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
 */
class UnscentedKalmanFilter(override val uid: String)
  extends KalmanTransformer[
    UnscentedKalmanStateCompute,
    UnscentedKalmanStateSpec,
    UnscentedKalmanFilter]
  with HasProcessFunction with HasMeasurementFunction with AdaptiveNoiseParams with SigmaPointsParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("unscentedKalmanFilter"))

  protected val defaultStateKey: String = "filter.unscentedKalmanFilter.defaultStateKey"

  /**
   * Set process function which governs state transition. It should accept the current stateVector
   * and processModel as arguments, and should output a vector of size (stateSize)
   * @group setParam
   */
  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  /**
   * Set measurement function which maps state to measurements. It should accept the current state vector
   * and measurementModel matrix as arguments, and should output a measurement vector of size (measurementSize)
   * @group setParam
   */
  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  /**
   * Set sigma point sampling algorithm for unscented transformation. Allowed values are;
   *
   * - 'merwe' (default) - E. Merwe (2000) The Unscented kalman filter for nonlinear estimation
   * - 'julier' - S. Julier (1997) A new extension to kalman filter to nonlinear systems
   * @group setParam
   */
  def setSigmaPoints(value: String): this.type = set(sigmaPoints, value)

  /**
   * Set alpha parameter for merwe algorithm
   *
   * Default is 0.3.
   * @group setParam
   */
  def setMerweAlpha(value: Double): this.type = set(merweAlpha, value)

  /**
   * Set beta parameter for merwe algorithm.
   *
   * Default is 2.0, tuned for gaussian noise.
   * @group setParam
   */
  def setMerweBeta(value: Double): this.type = set(merweBeta, value)

  /**
   * Set kappa parameter for merwe algorithm
   *
   * Default is 0.1. Suggested value is (3 - stateSize)
   * @group setParam
   */
  def setMerweKappa(value: Double): this.type = set(merweKappa, value)

  /**
   * Set kappa parameter for julier algorithm
   *
   * Default is 1.0.
   * @group setParam
   */
  def setJulierKappa(value: Double): this.type = set(julierKappa, value)

  /**
   * Enable adaptive process noise according to B. Zheng (2018) RAUKF paper
   * @group setParam
   */
  def setEnableAdaptiveProcessNoise: this.type = set(adaptiveProcessNoise, true)

  /**
   * Sets a lower bound for sigma point sampling. Lower bound is enforced with 'clipping'.
   *
   * By default there is no lower bound.
   * @group setParam
   */
  def setSigmaPointLowerBound(value: Vector): this.type = set(sigmaPointLowerBound, value)

  /**
   * Sets an upper bound for sigma point sampling. Upper bound is enforced with 'clipping'.
   *
   * By default there is no upper bound.
   * @group setParam
   */
  def setSigmaPointUpperBound(value: Vector): this.type = set(sigmaPointUpperBound, value)

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): UnscentedKalmanFilter =  {
    val that = new UnscentedKalmanFilter()
    copyValues(that, extra)
  }

  protected def stateUpdateSpec: UnscentedKalmanStateSpec = new UnscentedKalmanStateSpec(
    getFadingFactor,
    getSigmaPoints(getLinalgOptions),
    getProcessFunctionOpt,
    getMeasurementFunctionOpt,
    outputResiduals,
    getAdaptiveNoiseParamSet,
    getSlidingLikelihoodWindow,
    getMultiStepPredict
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
    val likelihoodWindow: Int,
    val multiStepPredict: Int)
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

    val sigmaPoints = sigma.sigmaPoints(state.state)

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

    val newState = sigma.unscentedTransform(
      state.state.mean.size,
      stateSigmaPoints,
      processNoise.toDense,
      fadingFactorSquare)

    KalmanState(state.stateIndex + 1, newState, state.residual, state.processNoise, state.slidingLoglikelihood)
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

    /* Different from E. Merwe (2000) paper, sigma points are re-calculated from predicted state rather than
    estimated state from previous time step. Produces marginally better estimates.*/
    val stateSigmaPoints = sigma.sigmaPoints(state.state)
    val measurementModel = process.measurementModel.get

    // Default measurement function is measurementModel * state
    val measurementFunction = measurementFunc.getOrElse(
      (in: Vector, model: Matrix) => model.multiply(in))
    val measurementNoise = process.measurementNoise.get.toDense

    // Propagate state through measurement function & perform unscented transform
    val measurementSigmaPoints = stateSigmaPoints
      .map(x => measurementFunction(x, measurementModel).toDense)
    val estimateDist = sigma
      .unscentedTransform(state.state.mean.size, measurementSigmaPoints, measurementNoise, 1.0)
    val (estimateMean, estimateCov) = (estimateDist.mean.toDense, estimateDist.covariance.toDense)
    val fadingFactorSquare = scala.math.pow(fadingFactor, 2)

    val crossCov = DenseMatrix.zeros(state.state.mean.size, process.measurement.get.size)
    stateSigmaPoints.zip(measurementSigmaPoints).zipWithIndex.foreach {
      case ((stateSigma, measurementSigma), i) => {

        val stateResidual = stateSigma.copy
        BLAS.axpy(-1.0, state.state.mean, stateResidual)
        val measurementResidual = measurementSigma.copy
        BLAS.axpy(-1.0, estimateMean, measurementResidual)
        val cw = sigma.covWeights(state.state.mean.size)
        BLAS.dger(
          cw(i) * fadingFactorSquare,
          stateResidual, measurementResidual, crossCov)
      }
    }

    val gain = crossCov.multiply(LinalgUtils.pinv(estimateCov)(sigma.ops))
    val residual = process.measurement.get.copy.toDense
    BLAS.axpy(-1.0, estimateMean, residual)
    val newMean = state.state.mean.toDense.copy
    BLAS.gemv(1.0, gain, residual, 1.0, newMean)
    val covUpdate = gain.multiply(estimateCov).multiply(gain.transpose)
    val newCov = DenseMatrix.zeros(state.state.covariance.numRows, state.state.covariance.numCols)
    BLAS.axpy(fadingFactorSquare, state.state.covariance.toDense, newCov)
    BLAS.axpy(-1.0, covUpdate, newCov)

    val resDist = if (storeResidual) Some(MultivariateGaussianDistribution(residual, estimateCov)) else None

    val processNoise = getAdaptiveProcessNoise(
      state.processNoise.getOrElse(process.processNoise.get).toDense,
      residual,
      estimateCov,
      gain
    )
    val ll = updateSlidingLikelihood(state.slidingLoglikelihood, likelihoodWindow, resDist)
    val newDist = MultivariateGaussianDistribution(newMean, newCov)
    KalmanState(state.stateIndex, newDist, resDist, processNoise, ll)
  }
}

/**
 * Base trait for sigma point algorithms for performing unscented transform
 */
private[filter] trait SigmaPoints extends Serializable {

  def meanWeights(stateSize: Int): DenseVector

  def covWeights(stateSize: Int): DenseVector

  val lbound: Option[Vector]

  val ubound: Option[Vector]

  val ops: LinalgOptions

  protected def applyBounds(in: DenseVector): DenseVector = {
    val arr = in.values
    lbound match {
      case Some(vec) => vec.toArray.zipWithIndex.foreach { case(bound, i)  =>
        if (arr(i) < bound) { arr(i) = bound }
      }
      case None =>
    }
    ubound match {
      case Some(vec) => vec.toArray.zipWithIndex.foreach { case(bound, i) =>
        if (arr(i) > bound) { arr(i) = bound }
      }
      case None =>
    }
    new DenseVector(arr)
  }

  // Not stored as a matrix due to column wise operations
  def sigmaPoints(dist: MultivariateGaussianDistribution): List[DenseVector]

  def unscentedTransform(
    stateSize: Int,
    sigmaPoints: List[DenseVector],
    noise: DenseMatrix,
    fadeFactor: Double): MultivariateGaussianDistribution = {

    val mw = meanWeights(stateSize)
    val newMean = new DenseVector(Array.fill(noise.numCols) {0.0})
    sigmaPoints.zipWithIndex.foreach {
      case(sigma, i) => {
        BLAS.axpy(mw(i), sigma, newMean)
      }
    }

    val cw = covWeights(stateSize)
    val newCov = noise.copy
    sigmaPoints.zipWithIndex.foreach { case(sigma, i) =>
      val residual = sigma.copy
      BLAS.axpy(-1.0, newMean, residual)
      BLAS.dger(cw(i) * fadeFactor, residual, residual, newCov)
    }
    MultivariateGaussianDistribution(newMean, newCov)
  }
}


private[filter] class JulierSigmaPoints(
    val kappa: Double,
    val lbound: Option[Vector],
    val ubound: Option[Vector],
    val ops: LinalgOptions) extends SigmaPoints {

  private def initConst(stateSize: Int) = 0.5/(stateSize + kappa)

  def meanWeights(stateSize: Int): DenseVector = {
    val weights = Array.fill(2 * stateSize + 1) { initConst(stateSize) }
    weights(0) = kappa / (kappa + stateSize)
    new DenseVector(weights)
  }

  def covWeights(stateSize: Int): DenseVector = meanWeights(stateSize)

  def sigmaPoints(dist: MultivariateGaussianDistribution): List[DenseVector] = {
    val mean = dist.mean.toDense
    val cov = dist.covariance.toDense
    val covUpdate = DenseMatrix.zeros(cov.numRows, cov.numCols)
    BLAS.axpy(kappa + mean.size, cov, covUpdate)
    val sqrt = LinalgUtils.sqrt(covUpdate)(ops)

    val (pos, neg) = sqrt.rowIter.foldLeft((List(mean), List[DenseVector]())) {
      case (coeffs, right) => {
        val meanPos = mean.copy
        BLAS.axpy(1.0, right, meanPos)
        val meanNeg = mean.copy
        BLAS.axpy(-1.0, right, meanNeg)
        (applyBounds(meanPos)::coeffs._1, applyBounds(meanNeg)::coeffs._2)
      }
    }
    pos.reverse:::neg.reverse
  }
}



private[filter] class MerweSigmaPoints(
    val alpha: Double,
    val beta: Double,
    val kappa: Double,
    val lbound: Option[Vector],
    val ubound: Option[Vector],
    val ops: LinalgOptions) extends SigmaPoints {

  private def lambda(stateSize: Int) = pow(alpha, 2) * (stateSize + kappa) - stateSize

  private def initConst(stateSize: Int) = 0.5 / (stateSize + lambda(stateSize))

  def meanWeights(stateSize: Int): DenseVector = {
    val weights = Array.fill(2 * stateSize + 1) { initConst(stateSize) }
    weights(0) = lambda(stateSize) / (stateSize + lambda(stateSize))
    new DenseVector(weights)
  }

  def covWeights(stateSize: Int): DenseVector = {
    val weights = Array.fill(2 * stateSize + 1) { initConst(stateSize) }
    weights(0) = lambda(stateSize) / (stateSize + lambda(stateSize)) + (1 - pow(alpha, 2) + beta)
    new DenseVector(weights)
  }

  def sigmaPoints(dist: MultivariateGaussianDistribution): List[DenseVector] = {
    val covUpdate = DenseMatrix.zeros(dist.covariance.numRows, dist.covariance.numCols)
    BLAS.axpy(lambda(dist.mean.size) + dist.mean.size, dist.covariance.toDense, covUpdate)
    val sqrt = LinalgUtils.sqrt(covUpdate)(ops)

    val (pos, neg) = sqrt.rowIter.foldLeft((List(dist.mean.toDense), List[DenseVector]())) {
      case (coeffs, right) => {
        val meanPos = dist.mean.toDense.copy
        BLAS.axpy(1.0, right, meanPos)
        val meanNeg = dist.mean.toDense.copy
        BLAS.axpy(-1.0, right, meanNeg)
        (applyBounds(meanPos)::coeffs._1, applyBounds(meanNeg)::coeffs._2)
      }
    }
    pos.reverse:::neg.reverse
  }

}


private[filter] trait HasMerweAlpha extends Params {

  /**
   * Alpha parameter for merwe sigma point algorithm. Advised to be between 0 and 1.
   *
   * Default is 0.3
   *
   * @group param
   */
  final val merweAlpha: DoubleParam = new DoubleParam(
    this,
    "merweAlpha",
    "Alpha parameter for merwe sigma point algorithm. Advised to be between 0 and 1"
  )

  setDefault(merweAlpha, 0.3)

  /**
   * Getter for merwe alpha parameter
   *
   * @group getParam
   */
  final def getMerweAlpha: Double = $(merweAlpha)
}


private[filter] trait HasMerweBeta extends Params {

  /**
   * Beta parameter for merwe sigma point algorithm. 2.0 is advised for gaussian noise
   *
   * Default is 2.0
   *
   * @group param
   */
  final val merweBeta: DoubleParam = new DoubleParam(
    this,
    "merweBeta",
    "Beta parameter for merwe sigma point algorithm. 2.0 is advised for gaussian noise"
  )

  setDefault(merweBeta, 2.0)

  /**
   * Getter for beta parameter
   *
   * @group getParam
   */
  final def getMerweBeta: Double = $(merweBeta)
}


private[filter] trait HasMerweKappa extends Params {

  /**
   * Kappa parameter for merwe sigma point algorithm. Advised value is (3 - stateSize)
   *
   * Default is 0.1
   *
   * @group param
   */
  final val merweKappa: DoubleParam = new DoubleParam(
    this,
    "merweKappa",
    "Kappa parameter for merwe sigma point algorithm. Advised value is (3 - stateSize)"
  )

  setDefault(merweKappa, 0.1)

  /**
   * Getter for merwe kappa parameter
   *
   * @group getParam
   */
  final def getMerweKappa: Double = $(merweKappa)
}


private[filter] trait HasJulierKappa extends Params {

  /**
   * Kappa parameter for julier sigma point algorithm.
   *
   * Default is 1.0
   *
   * @group param
   */
  final val julierKappa: DoubleParam = new DoubleParam(
    this,
    "julierKappa",
    "Kappa parameter for julier sigma point algorithm."
  )

  setDefault(julierKappa, 1.0)

  /**
   * Getter for julier kappa parameter
   *
   * @group getParam
   */
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

  /**
   * Param for enabling adaptive process noise.
   *
   * Disabled by default
   *
   * @group param
   */
  final val adaptiveProcessNoise: BooleanParam = new BooleanParam(
    this,
    "adaptiveProcessNoise",
    "Enable adaptive process noise according to B. Zheng(2018) RAUKF paper"
  )
  setDefault(adaptiveProcessNoise, false)

  /**
   * Getter for adaptive process noise flag
   *
   * @group getParam
   */
  final def getAdaptiveProcessNoise: Boolean = $(adaptiveProcessNoise)
}

private[filter] trait HasAdaptiveProcessNoiseThreshold extends Params {

  /**
   * Threshold for activating adaptive process noise, measured as mahalanobis distance from residual and
   * its covariance.
   *
   * Default is 2.0
   *
   * @group param
   */
  final val adaptiveProcessNoiseThreshold: DoubleParam = new DoubleParam(
    this,
    "adaptiveProcessNoiseThreshold",
    "Threshold for activating adaptive process noise, measured as mahalanobis distance from residual and" +
    "its covariance."
  )

  setDefault(adaptiveProcessNoiseThreshold, 2.0)

  /**
   * Getter for adaptive process noise threshold
   *
   * @group getParam
   */
  final def getAdaptiveProcessNoiseThreshold: Double = $(adaptiveProcessNoiseThreshold)
}


private[filter] trait HasAdaptiveProcessNoiseLambda extends Params {

  /**
   * Weight factor controlling the stability of noise updates. Should be between 0 and 1
   *
   * Default is 0.9
   *
   * @group param
   */
  final val adaptiveProcessNoiseLambda: DoubleParam = new DoubleParam(
    this,
    "adaptiveProcessNoiseLambda",
    "Weight factor controlling the stability of noise updates. Should be between 0 and 1"
  )

  setDefault(adaptiveProcessNoiseLambda, 0.9)

  /**
   * Getter for adaptive process noise lambda parameter
   *
   * @group getParam
   */
  final def getAdaptiveProcessNoiseLambda: Double = $(adaptiveProcessNoiseLambda)
}

private[filter] trait HasAdaptiveProcessNoiseAlpha extends Params {

  /**
   * Weight factor controlling the sensitivity of noise updates. Should be greater than 0.0
   * Large values give more influence to lambda factor.
   *
   * @group param
   */
  final val adaptiveProcessNoiseAlpha: DoubleParam = new DoubleParam(
    this,
    "adaptiveProcessNoiseAlpha",
    "Weight factor controlling the sensitivity of noise updates. Should be greater than 0.0" +
    "Large values give more influence to lambda factor"
  )

  setDefault(adaptiveProcessNoiseAlpha, 1.0)

  /**
   * Getter for adaptive process noise alpha parameter
   *
   * @group getParam
   */
  final def getAdaptiveProcessNoiseAlpha: Double = $(adaptiveProcessNoiseAlpha)
}


private[filter] trait HasSigmaPointLowerBound extends Params {

  /**
   * Lower bound vector for sigma point sampling. If set, generated sigma point samples will be
   * bounded. If state transition and measurement functions also respect these bounds, then the estimated state
   * will be bounded for all measurements.
   *
   * @group param
   */
  final val sigmaPointLowerBound: Param[Vector] = new Param[Vector](
    this,
    "sigmaPointLowerBound",
    "Lower bound vector for sigma point sampling. If set, generated sigma point samples will be" +
    "bounded. If state transition and measurement functions also respect these bounds, then the estimated state" +
    "will be bounded for all measurements."
  )

  /**
   * Getter for sigma point lower bound
   *
   * @group getParam
   */
  final def getSigmaPointLowerBound: Option[Vector] = get(sigmaPointLowerBound)

}

private[filter] trait HasSigmaPointUpperBound extends Params {

  /**
   * Upper bound vector for sigma point sampling. If set, generated sigma point samples will be
   * bounded. If state transition and measurement functions also respect these bounds, then the estimated state
   * will be bounded for all measurements.
   *
   * @group param
   */
  final val sigmaPointUpperBound: Param[Vector] = new Param[Vector](
    this,
    "sigmaPointUpperBound",
    "Upper bound vector for sigma point sampling. If set, generated sigma point samples will be" +
      "bounded. If state transition and measurement functions also respect these bounds, then the estimated state" +
      "will be bounded for all measurements."
  )

  /**
   * Getter for sigma point upper bound
   *
   * @group getParam
   */
  final def getSigmaPointUpperBound: Option[Vector] = get(sigmaPointUpperBound)

}

/**
 * Trait for parameters of sigma point algorithms.
 */
private[filter] trait SigmaPointsParams extends HasMerweAlpha with HasMerweBeta with HasMerweKappa with HasJulierKappa
  with HasSigmaPointLowerBound with HasSigmaPointUpperBound {


  /**
   * Parameter for choosing sigma point sampling algorithm. Options are 'merwe' and 'julier'
   *
   * @group param
   */
  final val sigmaPoints: Param[String] = new Param[String](
    this,
    "sigmaPoints",
    "Sigma point sampling algorithm, options are 'merwe' and 'julier'"
  )
  setDefault(sigmaPoints, "merwe")

  /**
   * Getter for sigma point algorithm helper class
   *
   * @group getParam
   */
  final def getSigmaPoints(ops: LinalgOptions): SigmaPoints = {
    $(sigmaPoints) match {
      case "merwe" => new MerweSigmaPoints(
        $(merweAlpha), $(merweBeta), $(merweKappa), getSigmaPointLowerBound, getSigmaPointUpperBound, ops)
      case "julier" => new JulierSigmaPoints(
        $(julierKappa), getSigmaPointLowerBound, getSigmaPointUpperBound, ops)
      case _ => throw new Exception("Unsupported sigma point algorithm")
    }
  }
}


/**
 * Companion object of UnscentedKalmanFilter for read/write
 */
object UnscentedKalmanFilter extends DefaultParamsReadable[UnscentedKalmanFilter] {

  override def load(path: String): UnscentedKalmanFilter = super.load(path)
}