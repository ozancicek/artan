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
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS


/**
 * Cubature Kalman Filter (CKF), implemented with a stateful spark Transformer for running parallel filters /w spark
 * dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
 * spark transformations, which can be used in both streaming and batch applications. CKF is similar to UKF,
 * it could be seen as a special case of UKF with good parameters for most general problems.
 *
 * In addition to Linear Kalman Filter parameters, following functions
 * can be specified assuming a state (x_k) with size n_s, and measurements (z_k) with size n_m;
 *
 * - f(x_k, F_k), process function for state transition. x_k is state vector and F_k is process model.
 *   Should output a vector with size (n_s)
 *
 * - h(x_k, H_k), measurement function. Should output a vector with size (n_m)
 *
 *
 * CKF will predict & estimate the state according to following equations;
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
 * @param stateSize size of the state vector
 * @param measurementSize size of the measurement vector
 */
class CubatureKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[
    CubatureKalmanStateCompute,
    CubatureKalmanStateSpec,
    CubatureKalmanFilter]
  with HasProcessFunction with HasMeasurementFunction {

  def this(
    stateSize: Int,
    measurementSize: Int) = {
    this(stateSize, measurementSize, Identifiable.randomUID("cubatureKalmanFilter"))
  }

  protected val defaultStateKey: String = "filter.cubatureKalmanFilter.defaultStateKey"

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
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): CubatureKalmanFilter =  {
    val that = new CubatureKalmanFilter(stateSize, measurementSize)
    copyValues(that, extra)
  }

  protected def stateUpdateSpec: CubatureKalmanStateSpec = new CubatureKalmanStateSpec(
    getFadingFactor,
    new CubaturePoints(stateSize),
    getProcessFunctionOpt,
    getMeasurementFunctionOpt,
    outputResiduals,
    getSlidingLikelihoodWindow
  )
}

/**
 * Function spec for CKF.
 */
private[filter] class CubatureKalmanStateSpec(
    val fadingFactor: Double,
    val cubature: CubaturePoints,
    val processFunction: Option[(Vector, Matrix) => Vector],
    val measurementFunction: Option[(Vector, Matrix) => Vector],
    val storeResidual: Boolean,
    val likelihoodWindow: Int)
  extends KalmanStateUpdateSpec[CubatureKalmanStateCompute] {

  val kalmanCompute = new CubatureKalmanStateCompute(
    fadingFactor,
    cubature,
    processFunction,
    measurementFunction)
}

/**
 * Class responsible for calculating CKF updates
 */
private[filter] class CubatureKalmanStateCompute(
    fadingFactor: Double,
    cubature: CubaturePoints,
    processFunc: Option[(Vector, Matrix) => Vector],
    measurementFunc: Option[(Vector, Matrix) => Vector]) extends KalmanStateCompute {

  def predict(
    state: KalmanState,
    process: KalmanInput): KalmanState = {

    val cubaturePoints = cubature.cubaturePoints(state.state)

    val processModel = process.processModel.get
    val processFunction = processFunc.getOrElse(
      (in: Vector, model: Matrix) => processModel.multiply(in))
    val processNoise = state.processNoise.getOrElse(process.processNoise.get)

    val stateCubaturePoints = cubaturePoints.map { cubatures =>
      val newCubatures = processFunction(cubatures, processModel).toDense
      (process.control, process.controlFunction) match {
        case (Some(vec), Some(func)) => BLAS.gemv(1.0, func, vec, 1.0, newCubatures)
        case _ =>
      }
      newCubatures
    }

    val newDist = cubature.cubatureTransform(
      stateCubaturePoints,
      processNoise.toDense,
      scala.math.pow(fadingFactor, 2))
    KalmanState(
      state.stateIndex + 1, newDist,
      state.residual, state.processNoise, state.slidingLoglikelihood)
  }

  private def estimateCrossCovariance(
    stateCubaturePoints: List[DenseVector],
    stateMean: DenseVector,
    measurementCubaturePoints: List[DenseVector],
    measurementMean: DenseVector): DenseMatrix = {

    val crossCov = DenseMatrix.zeros(stateMean.size, measurementMean.size)
    stateCubaturePoints.zip(measurementCubaturePoints).foreach { case(stateCub, measCub) =>
      BLAS.dger(1.0/stateCubaturePoints.size, stateCub, measCub, crossCov)
      BLAS.dger(-1.0/stateCubaturePoints.size, stateMean, measurementMean, crossCov)
    }
    crossCov
  }

  def estimate(
    state: KalmanState,
    process: KalmanInput,
    storeResidual: Boolean,
    likelihoodWindow: Int): KalmanState = {

    val (stateMean, stateCov) = (state.state.mean.toDense, state.state.covariance.toDense)

    val stateCubaturePoints = cubature.cubaturePoints(state.state)
    val measurementModel = process.measurementModel.get

    // Default measurement function is measurementModel * state
    val measurementFunction = measurementFunc.getOrElse(
      (in: Vector, model: Matrix) => model.multiply(in))
    val measurementNoise = process.measurementNoise.get.toDense

    // Propagate state through measurement function & perform cubature transform
    val measurementCubaturePoints = stateCubaturePoints
      .map(x => measurementFunction(x, measurementModel).toDense)

    val estimateDist = cubature
      .cubatureTransform(measurementCubaturePoints, measurementNoise, 1.0)
    val (estimateMean, estimateCov) = (estimateDist.mean.toDense, estimateDist.covariance.toDense)

    val crossCov = estimateCrossCovariance(stateCubaturePoints, stateMean, measurementCubaturePoints, estimateMean)

    val gain = crossCov.multiply(LinalgUtils.pinv(estimateCov))

    val residual = process.measurement.get.copy.toDense
    BLAS.axpy(-1.0, estimateMean, residual)
    val newMean = stateMean.copy
    BLAS.gemv(1.0, gain, residual, 1.0, newMean)

    val covUpdate = gain.multiply(estimateCov).multiply(gain.transpose)
    val newCov = DenseMatrix.zeros(stateCov.numRows, stateCov.numCols)
    BLAS.axpy(1.0, stateCov, newCov)
    BLAS.axpy(-1.0, covUpdate, newCov)

    val resDist = if (storeResidual) Some(MultivariateGaussianDistribution(residual, estimateCov)) else None
    val ll = updateSlidingLikelihood(state.slidingLoglikelihood, likelihoodWindow, resDist)

    val newDist = MultivariateGaussianDistribution(newMean, newCov)
    KalmanState(
      state.stateIndex, newDist, resDist, state.processNoise, ll)
  }
}

/**
 * Class for sampling cubature points
 */
private[filter] class CubaturePoints(val stateSize: Int) extends Serializable {

  def rotateRight[A](seq: Seq[A], i: Int): Seq[A] = {
    val size = seq.size
    val (first, last) = seq.splitAt(size - (i % size))
    last ++ first
  }

  def genSymmetricVectors(weight: Double): List[DenseVector] = {
    val weights = weight :: List.fill(stateSize - 1) {0.0}
    (0 until stateSize).toList.map {i =>
      new DenseVector(rotateRight(weights, i).toArray)
    }
  }

  lazy val cubatureVectors: List[DenseVector] = {
    val weight = scala.math.sqrt(stateSize)
    genSymmetricVectors(weight) ++ genSymmetricVectors(-weight)
  }

  def cubaturePoints(distribution: MultivariateGaussianDistribution): List[DenseVector] = {
    val sqrtCov = LinalgUtils.sqrt(distribution.covariance.toDense)
    cubatureVectors.map { cubVec =>
      val point = distribution.mean.toDense.copy
      BLAS.gemv(1.0, sqrtCov, cubVec, 1.0, point)
      point
    }
  }

  def cubatureTransform(
    cubaturePoints: List[DenseVector],
    noise: DenseMatrix,
    weight: Double): MultivariateGaussianDistribution = {

    val newMean = new DenseVector(Array.fill(noise.numCols) {0.0})
    cubaturePoints.foreach { point =>
      BLAS.axpy(1.0/cubaturePoints.size, point, newMean)
    }

    val newCov = noise.copy

    cubaturePoints.foreach { cubVec =>
      BLAS.dger(weight/cubaturePoints.size, cubVec, cubVec, newCov)
      BLAS.dger( -weight/cubaturePoints.size, newMean, newMean, newCov)
    }

    MultivariateGaussianDistribution(newMean, newCov)
  }
}