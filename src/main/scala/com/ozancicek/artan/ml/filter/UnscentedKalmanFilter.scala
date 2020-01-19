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

import com.ozancicek.artan.ml.linalg.LinalgUtils
import com.ozancicek.artan.ml.state.{KalmanState, KalmanInput}
import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Matrix}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import scala.math.pow

/**
 * Unscented Kalman Filter (UKF), implemented with a stateful spark Transformer for running parallel filters /w spark
 * dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
 * spark transormations, which can be used in both streaming and batch applications.
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
  with HasProcessFunction with HasMeasurementFunction with SigmaPointsParams {

  def this(
    measurementSize: Int,
    stateSize: Int) = {
    this(measurementSize, stateSize, Identifiable.randomUID("unscentedKalmanFilter"))
  }

  protected val defaultStateKey: String = "filter.unscentedKalmanFilter"

  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  def setSigmaPoints(value: String): this.type = set(sigmaPoints, value)

  def setMerweAlpha(value: Double): this.type = set(merweAlpha, value)

  def setMerweBeta(value: Double): this.type = set(merweBeta, value)

  def setMerweKappa(value: Double): this.type = set(merweKappa, value)

  def setJulierKappa(value: Double): this.type = set(julierKappa, value)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = filter(dataset)

  protected def stateUpdateSpec: UnscentedKalmanStateSpec = new UnscentedKalmanStateSpec(
    getInitialState,
    getInitialCovariance,
    getFadingFactor,
    getSigmaPoints,
    getProcessFunctionOpt,
    getMeasurementFunctionOpt,
    outputResiduals
  )
}


private[ml] class UnscentedKalmanStateSpec(
    val stateMean: Vector,
    val stateCov: Matrix,
    val fadingFactor: Double,
    val sigma: SigmaPoints,
    val processFunction: Option[(Vector, Matrix) => Vector],
    val measurementFunction: Option[(Vector, Matrix) => Vector],
    val storeResidual: Boolean)
  extends KalmanStateUpdateSpec[UnscentedKalmanStateCompute] {

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

  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState = {

    val sigmaPoints = sigma.sigmaPoints(state.state.toDense, state.stateCovariance.toDense)

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
      state.stateIndex + 1, stateMean, stateCov, state.residual, state.residualCovariance)
  }

  def estimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean): KalmanState = {

    val (stateMean, stateCov) = (state.state.toDense, state.stateCovariance.toDense)
    val stateSigmaPoints = sigma.sigmaPoints(stateMean, stateCov)
    val measurementModel = process.measurementModel.get
    val measurementFunction = measurementFunc.getOrElse(
      (in: Vector, model: Matrix) => model.multiply(in))
    val measurementNoise = process.measurementNoise.get.toDense
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

    KalmanState(
      state.stateIndex, newMean, newCov, res, resCov)
  }
}


private[filter] trait SigmaPoints extends Serializable {

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


private[filter] trait HasMerweAlpha extends Params {

  final val merweAlpha: DoubleParam = new DoubleParam(
    this,
    "merweAlpha",
    "merwe alpha"
  )

  setDefault(merweAlpha, 0.3)

  final def getMerweAlpha: Double = $(merweAlpha)
}


private[filter] trait HasMerweBeta extends Params {

  final val merweBeta: DoubleParam = new DoubleParam(
    this,
    "merweBeta",
    "merwe beta"
  )

  setDefault(merweBeta, 2.0)

  final def getMerweBeta: Double = $(merweBeta)
}


private[filter] trait HasMerweKappa extends Params {

  final val merweKappa: DoubleParam = new DoubleParam(
    this,
    "merweKappa",
    "merwe kappa"
  )

  setDefault(merweKappa, 0.1)

  final def getMerweKappa: Double = $(merweKappa)
}


private[filter] trait HasJulierKappa extends Params {

  final val julierKappa: DoubleParam = new DoubleParam(
    this,
    "julierKappa",
    "julier kappa"
  )

  setDefault(julierKappa, 1.0)

  final def getJulierKappa: Double = $(julierKappa)
}


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
      case _ => throw new Exception("Unsupported sigma point option")
    }
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
