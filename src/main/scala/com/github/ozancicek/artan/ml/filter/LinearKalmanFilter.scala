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
import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS


/**
 * Linear Kalman Filter, implemented with a stateful spark Transformer for running parallel filters /w spark
 * dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
 * spark transformations, which can be used in both streaming and batch applications.
 *
 * Assuming a state (x_k) with size n_s, and measurements (z_k) with size n_m, following parameters should be specified;
 *
 * - F_k, process model, matrix with dimensions (n_s, n_s)
 * - H_k, measurement model, matrix with dimensions (n_s, n_m)
 * - Q_k, process noise covariance, matrix with dimensions (n_s, n_s)
 * - R_k, measurement noise covariance, matrix with dimensions (n_m, n_m)
 * - B_k, optional control model, matrix with dimensions (n_s, n_control)
 * - u_k, optional control vector, vector with size (n_control)
 *
 * Linear Kalman Filter will predict & estimate the state according to following equations
 *
 * State prediction:
 *  x_k = F_k * x_k-1 + B_k * u_k + w_k
 *
 * Measurement update:
 *  z_k = H_k * x_k + v_k
 *
 * Where v_k and w_k are noise vectors drawn from zero mean, Q_k and R_k covariance distributions.
 *
 * The default values of system matrices will not give you a functioning filter, but they will be initialized
 * with reasonable values given the state and measurement sizes. All of the inputs to the filter can
 * be specified with a dataframe column which will allow you to have different value across measurements/filters,
 * or you can specify a constant value across all measurements/filters.
 *
 * @param stateSize size of the state vector
 * @param measurementSize size of the measurement vector
 */
class LinearKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[LinearKalmanStateCompute, LinearKalmanStateSpec, LinearKalmanFilter] {

  def this(
    stateSize: Int,
    measurementSize: Int) = {
    this(stateSize, measurementSize, Identifiable.randomUID("linearKalmanFilter"))
  }

  protected val defaultStateKey: String = "filter.linearKalmanFilter.defaultStateKey"

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): LinearKalmanFilter =  {
    val that = new LinearKalmanFilter(stateSize, measurementSize)
    copyValues(that, extra)
  }

  protected def stateUpdateSpec: LinearKalmanStateSpec = new LinearKalmanStateSpec(
    getFadingFactor,
    outputResiduals,
    getSlidingLikelihoodWindow,
    getMultiStepPredict
  )
}

/**
 * Function spec for updating linear kalman state
 *
 * @param fadingFactor Factor weighting for recent measurements, >= 1.0
 * @param storeResidual Boolean flag to store residuals in the state
 */
private[filter] class LinearKalmanStateSpec(
    val fadingFactor: Double,
    val storeResidual: Boolean,
    val likelihoodWindow: Int,
    val multiStepPredict: Int)
  extends KalmanStateUpdateSpec[LinearKalmanStateCompute] {

  val kalmanCompute = new LinearKalmanStateCompute(fadingFactor)
}

/**
 * Implements the prediction & estimation cycle of linear kalman filter. Assuming the state at time step
 * k-1 is defined by state vector x_k-1 and covariance matrix P_k-1;
 *
 * State prediction equations:
 *  x_k = F * x_k-1 + B * u
 *  P_k = a * F * P_k-1 * F.T + Q
 *
 * Measurement update equations:
 *  r = z_k - H * x_k
 *  K = P_k * H.T * inv(H * P_k * H.T + R)
 *  x_k = x_k + K * r
 *  P_k = (I - K * H) * P_k * (I - K * H).T + K * R * K.T
 *
 * Where;
 *
 * F: processModel
 * B: controlFunction
 * u: control
 * Q: processNoise
 * H: measurementModel
 * R: measurementNoise
 * a: fading factor squared
 *
 * @param fadingFactor Factor weighting for recent measurements, >= 1.0
 */
private[filter] class LinearKalmanStateCompute(
    val fadingFactor: Double) extends KalmanStateCompute {

  /* Apply the process model & predict the next state. */
  protected def progressStateMean(
    stateMean: DenseVector,
    processModel: DenseMatrix): DenseVector = {
    val mean = new DenseVector(Array.fill(stateMean.size) {0.0})
    BLAS.gemv(1.0, processModel, stateMean, 0.0, mean)
    mean
  }

  /* Apply the measurement model & calculate the residual*/
  protected def calculateResidual(
    stateMean: DenseVector,
    measurement: DenseVector,
    measurementModel: DenseMatrix): DenseVector = {
    val residual = measurement.copy
    BLAS.gemv(-1.0, measurementModel, stateMean, 1.0, residual)
    residual
  }

  protected def getProcessModel(
    stateMean: DenseVector,
    processModel: DenseMatrix): DenseMatrix = processModel

  protected def getMeasurementModel(
    stateMean: DenseVector,
    measurementModel: DenseMatrix): DenseMatrix = measurementModel

  protected def getProcessNoise(
    stateMean: DenseVector,
    processNoise: DenseMatrix): DenseMatrix = processNoise

  protected def getMeasurementNoise(
    stateMean: DenseVector,
    measurementNoise: DenseMatrix): DenseMatrix = measurementNoise

  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState = {

    // x_k = F * x_k-1
    val newMean = progressStateMean(
      state.state.mean.toDense, process.processModel.get.toDense)

    val pModel = getProcessModel(
      state.state.mean.toDense, process.processModel.get.toDense)

    // x_k += B * u
    (process.control, process.controlFunction) match {
      case (Some(vec), Some(func)) => BLAS.gemv(1.0, func, vec, 1.0, newMean)
      case _ =>
    }

    // covUpdate = a * F * P_k-1
    val covUpdate = DenseMatrix.zeros(pModel.numRows, state.state.covariance.numCols)
    val fadingFactorSquare = scala.math.pow(fadingFactor, 2)
    BLAS.gemm(fadingFactorSquare, pModel, state.state.covariance.toDense, 1.0, covUpdate)

    // P_k = covUpdate * F.T + Q
    val newCov = getProcessNoise(state.state.mean.toDense, process.processNoise.get.copy.toDense)
    BLAS.gemm(1.0, covUpdate, pModel.transpose, 1.0, newCov)

    val newDist = MultivariateGaussianDistribution(newMean, newCov)

    KalmanState(
      state.stateIndex + 1L,
      newDist,
      state.residual,
      state.processNoise,
      state.slidingLoglikelihood)
  }

  def estimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean,
    likelihoodWindow: Int): KalmanState = {

    // r = z_k - H * x_k
    val residual = calculateResidual(
      state.state.mean.toDense,
      process.measurement.get.toDense,
      process.measurementModel.get.toDense)

    val mModel = getMeasurementModel(
      state.state.mean.toDense,
      process.measurementModel.get.toDense)

    val mNoise = getMeasurementNoise(
      state.state.mean.toDense,
      process.measurementNoise.get.toDense)

    // speed = P * H.T
    val speed = state.state.covariance.multiply(mModel.transpose)
    // rCov = H * speed + R
    val residualCovariance = mNoise.copy
    BLAS.gemm(1.0, mModel, speed, 1.0, residualCovariance)

    // K = speed * inv(speed)
    val inverseUpdate = LinalgUtils.pinv(residualCovariance)
    val gain = DenseMatrix.zeros(speed.numRows, inverseUpdate.numCols)
    BLAS.gemm(1.0, speed, inverseUpdate, 1.0, gain)

    // x_k += K * r
    val estMean = state.state.mean.copy.toDense
    BLAS.gemv(1.0, gain, residual, 1.0, estMean)

    // ident = I - K * H
    val ident = DenseMatrix.eye(estMean.size)
    BLAS.gemm(-1.0, gain, mModel, 1.0, ident)

    // P_k = ident * P_k * ident.T + K * R * K.T
    val estCov = ident.multiply(state.state.covariance.toDense).multiply(ident.transpose)
    val noiseUpdate = gain.multiply(mNoise.toDense)
    BLAS.gemm(1.0, noiseUpdate, gain.transpose, 1.0, estCov)

    val resDist = if (storeResidual) Some(MultivariateGaussianDistribution(residual, residualCovariance)) else None

    val slidingll = updateSlidingLikelihood(state.slidingLoglikelihood, likelihoodWindow, resDist)

    val estDist = MultivariateGaussianDistribution(estMean, estCov)
    KalmanState(
      state.stateIndex,
      estDist,
      resDist,
      state.processNoise,
      slidingll)
  }
}