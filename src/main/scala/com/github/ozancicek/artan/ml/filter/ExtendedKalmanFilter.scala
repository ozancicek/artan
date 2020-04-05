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

import com.github.ozancicek.artan.ml.state.{KalmanInput, KalmanState}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS


/**
 * Extended Kalman Filter (EKF), implemented with a stateful spark Transformer for running parallel filters /w spark
 * dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
 * spark transformations, which can be used in both streaming and batch applications.
 *
 * Typically for nonlinear systems, it allows either of state transition and observation models to be
 * differentiable functions instead of matrices. It also allows speciying non-additive noise covariances for
 * state transition and measurements with noise jacobian matrices depending on state.
 *
 * All Linear Kalman Filter parameters are also valid for EKF. In addition to Linear Kalman
 * Filter parameters,following functions can be specified assuming a state (x_k) with size n_s,
 * and measurements (z_k) with size n_m;
 *
 * - f(x_k, F_k), process function for state transition. x_k is state vector and F_k is process model.
 *   Should output a vector with size (n_s)
 *
 * - f_j(x_k, F_k), process jacobian function for state transition. x_k is state vector and F_k is process model.
 *   Should output a matrix with dimensions (n_s, n_s)
 *
 * - q_j(x_k, Q_k). process noise jacobian function for non-additive noise. x_k is state vector and Q_k is process
 *   noise matrix with dimensions (n_noise, n_noise). Should output a matrix with dimensions (n_s, n_noise). The result
 *   of q_j * Q_k * q_j.T transformation should be (n_s, n_s)
 *
 * - h(x_k, H_k), measurement function. Should output a vector with size (n_m)
 *
 * - hj(x_j, H_k), measurement jacobian function. Should output a matrix with dimensions (n_s, n_m)
 *
 * - r_j(x_k, R_k). measurement noise jacobian function for non-additive noise. x_k is state vector and R_k is process
 *   noise matrix with dimensions (n_noise, n_noise). Should output a matrix with dimensions (n_s, n_noise). The result
 *   of q_j * Q_k * q_j.T transformation should be (n_s, n_s)
 *
 *
 * EKF will predict & estimate the state according to following equations
 *
 * State prediction:
 *  x_k = f(x_k-1, F_k) + B_k * u_k + w_k
 *
 * Measurement incorporation:
 *  z_k = h(x_k, H_k) + v_k
 *
 * Where v_k and w_k are noise vectors drawn from zero mean, q_j * Q_k * q_j.T and r_j * R_k *r_j.T covariance
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
class ExtendedKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[
    ExtendedKalmanStateCompute,
    ExtendedKalmanStateSpec,
    ExtendedKalmanFilter]
  with HasProcessFunction with HasProcessStateJacobian with HasProcessNoiseJacobian
  with HasMeasurementFunction with HasMeasurementStateJacobian with HasMeasurementNoiseJacobian {

  def this(
    stateSize: Int,
    measurementSize: Int) = {
    this(stateSize, measurementSize, Identifiable.randomUID("extendedKalmanFilter"))
  }
  protected val defaultStateKey: String = "filter.extendedKalmanFilter.defaultStateKey"

  /**
   * Set process function which governs state transition. It should accept the current stateVector
   * and processModel as arguments, and should output a vector of size (stateSize)
   */
  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  /**
   * Set process state jacobian function. It should accept the current stateVector
   * and processModel as arguments, and should output a matrix with dimensions (stateSize, stateSize)
   */
  def setProcessStateJacobian(value: (Vector, Matrix) => Matrix): this.type = set(processStateJacobian, value)

  /**
   * Set measurement function which maps state to measurements. It should accept the current state vector
   * and measurementModel matrix as arguments, and should output a measurement vector of size (measurementSize)
   */
  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  /**
   * Set measurement state jacobian function. It should accept the current stateVector
   * and processModel as arguments, and should output a matrix with dimensions (stateSize, measurementSize)
   */
  def setMeasurementStateJacobian(value: (Vector, Matrix) => Matrix): this.type = set(measurementStateJacobian, value)

  /**
   * Set process noise jacobian function. It should accept the current stateVector
   * and processNoise as arguments, and should output a matrix with dimensions (stateSize, noiseSize), where noiseSize
   * is the dimensions of square processNoise matrix.
   */
  def setProcessNoiseJacobian(value: (Vector, Matrix) => Matrix): this.type = set(processNoiseJacobian, value)

  /**
   * Set measurement noise jacobian function. It should accept the current stateVector
   * and measurementNoise as arguments, and should output a matrix with dimensions (measurementSize, noiseSize),
   * where noiseSize is the dimensions of square measurementNoise matrix.
   */
  def setMeasurementNoiseJacobian(value: (Vector, Matrix) => Matrix): this.type = set(measurementNoiseJacobian, value)

  override def copy(extra: ParamMap): ExtendedKalmanFilter =  {
    val that = new ExtendedKalmanFilter(stateSize, measurementSize)
    copyValues(that, extra)
  }

  protected def stateUpdateSpec: ExtendedKalmanStateSpec = new ExtendedKalmanStateSpec(
    getFadingFactor,
    getProcessFunctionOpt,
    getProcessStateJacobianOpt,
    getProcessNoiseJacobianOpt,
    getMeasurementFunctionOpt,
    getMeasurementStateJacobianOpt,
    getMeasurementNoiseJacobianOpt,
    outputResiduals,
    getSlidingLikelihoodWindow
  )
}

/**
 * Function spec for EKF
 */
private[filter] class ExtendedKalmanStateSpec(
    val fadingFactor: Double,
    val processFunction: Option[(Vector, Matrix) => Vector],
    val processStateJacobian: Option[(Vector, Matrix) => Matrix],
    val processNoiseJacobian: Option[(Vector, Matrix) => Matrix],
    val measurementFunction: Option[(Vector, Matrix) => Vector],
    val measurementStateJacobian: Option[(Vector, Matrix) => Matrix],
    val measurementNoiseJacobian: Option[(Vector, Matrix) => Matrix],
    val storeResidual: Boolean,
    val likelihoodWindow: Int)
  extends KalmanStateUpdateSpec[ExtendedKalmanStateCompute] {

  override def getOutputProcessModel(
    row: KalmanInput,
    state: KalmanState): Option[Matrix] = {
    row.processModel.map(m => kalmanCompute.getProcessModel(state.state.toDense, m.toDense))
  }

  override def getOutputProcessNoise(
    row: KalmanInput,
    state: KalmanState): Option[Matrix] = {
    row.processNoise.map(m => kalmanCompute.getProcessNoise(state.state.toDense, m.toDense))
  }

  override def getOutputMeasurementModel(
    row: KalmanInput,
    state: KalmanState): Option[Matrix] = {
    row.measurementModel.map(m => kalmanCompute.getMeasurementModel(state.state.toDense, m.toDense))
  }

  val kalmanCompute = new ExtendedKalmanStateCompute(
    fadingFactor,
    processFunction,
    processStateJacobian,
    processNoiseJacobian,
    measurementFunction,
    measurementStateJacobian,
    measurementNoiseJacobian)
}

/**
 * Class for calculating EKF updates. Extends the Linear Kalman Filter by overriding state transition & measurement
 * models with functions.
 */
private[filter] class ExtendedKalmanStateCompute(
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
