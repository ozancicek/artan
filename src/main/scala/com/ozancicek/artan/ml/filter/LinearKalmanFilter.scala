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
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{BLAS}
import org.apache.spark.sql._


class LinearKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[LinearKalmanStateCompute, LinearKalmanStateEstimator]
  with KalmanUpdateParams with HasInitialState with HasInitialCovariance with HasFadingFactor {

  def this(
    measurementSize: Int,
    stateSize: Int) = {
    this(measurementSize, stateSize, Identifiable.randomUID("linearKalmanFilter"))
  }

  def setInitialState(value: Vector): this.type = set(initialState, value)

  def setInitialCovariance(value: Matrix): this.type = set(initialCovariance, value)

  def setFadingFactor(value: Double): this.type = set(fadingFactor, value)

  def setProcessModel(value: Matrix): this.type = set(processModel, value)

  def setProcessNoise(value: Matrix): this.type = set(processNoise, value)

  def setMeasurementModel(value: Matrix): this.type = set(measurementModel, value)

  def setMeasurementNoise(value: Matrix): this.type = set(measurementNoise, value)

  def setStateKeyCol(value: String): this.type = set(stateKeyCol, value)

  def setMeasurementCol(value: String): this.type = set(measurementCol, value)

  def setProcessModelCol(value: String): this.type = set(processModelCol, value)

  def setProcessNoiseCol(value: String): this.type = set(processNoiseCol, value)

  def setMeasurementModelCol(value: String): this.type = set(measurementModelCol, value)

  def setMeasurementNoiseCol(value: String): this.type = set(measurementNoiseCol, value)

  def setControlCol(value: String): this.type = set(controlCol, value)

  def setControlFunctionCol(value: String): this.type = set(controlFunctionCol, value)

  def setCalculateLoglikelihood: this.type = set(calculateLoglikelihood, true)

  def setCalculateMahalanobis: this.type = set(calculateMahalanobis, true)

  override def copy(extra: ParamMap): LinearKalmanFilter = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = withExtraColumns(filter(dataset))

  protected def stateUpdateFunc: LinearKalmanStateEstimator = new LinearKalmanStateEstimator(
    getInitialState,
    getInitialCovariance,
    getFadingFactor
  )
}


private[filter] class LinearKalmanStateEstimator(
    val stateMean: Vector,
    val stateCov: Matrix,
    val fadingFactor: Double)
  extends KalmanStateUpdateFunction[LinearKalmanStateCompute] {

  val kalmanCompute = new LinearKalmanStateCompute(fadingFactor)
}


private[filter] class LinearKalmanStateCompute(
    val fadingFactor: Double) extends KalmanStateCompute {

  protected def progressStateMean(
    stateMean: DenseVector,
    processModel: DenseMatrix): DenseVector = {
    val mean = new DenseVector(Array.fill(stateMean.size) {0.0})
    BLAS.gemv(1.0, processModel, stateMean, 0.0, mean)
    mean
  }

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

    val newMean = progressStateMean(
      state.state.toDense, process.processModel.get.toDense)

    val pModel = getProcessModel(
      state.state.toDense, process.processModel.get.toDense)

    (process.control, process.controlFunction) match {
      case (Some(vec), Some(func)) => BLAS.gemv(1.0, func, vec, 1.0, newMean)
      case _ =>
    }

    val covUpdate = DenseMatrix.zeros(pModel.numRows, state.stateCovariance.numCols)
    val fadingFactorSquare = scala.math.pow(fadingFactor, 2)
    BLAS.gemm(fadingFactorSquare, pModel, state.stateCovariance.toDense, 1.0, covUpdate)

    val newCov = getProcessNoise(state.state.toDense, process.processNoise.get.copy.toDense)
    BLAS.gemm(1.0, covUpdate, pModel.transpose, 1.0, newCov)
    KalmanState(
      state.stateKey,
      state.stateIndex + 1L,
      newMean, newCov,
      state.residual,
      state.residualCovariance)
  }

  def estimate(
    state: KalmanState,
    process: KalmanInput): KalmanState = {

    val residual = calculateResidual(
      state.state.toDense,
      process.measurement.get.toDense,
      process.measurementModel.get.toDense)

    val mModel = getMeasurementModel(
      state.state.toDense,
      process.measurementModel.get.toDense)

    val mNoise = getMeasurementNoise(
      state.state.toDense,
      process.measurementNoise.get.toDense)

    val speed = state.stateCovariance.multiply(mModel.transpose)
    val noiseCov = mNoise.copy
    BLAS.gemm(1.0, mModel, speed, 1.0, noiseCov)

    val inverseUpdate = LinalgUtils.pinv(noiseCov)
    val gain = DenseMatrix.zeros(speed.numRows, inverseUpdate.numCols)
    BLAS.gemm(1.0, speed, inverseUpdate, 1.0, gain)

    val estMean = state.state.copy.toDense
    BLAS.gemv(1.0, gain, residual, 1.0, estMean)

    val ident = DenseMatrix.eye(estMean.size)
    BLAS.gemm(-1.0, gain, mModel, 1.0, ident)

    val estCov = ident.multiply(state.stateCovariance.toDense).multiply(ident.transpose)

    val noiseUpdate = gain.multiply(mNoise.toDense)

    BLAS.gemm(1.0, noiseUpdate, gain.transpose, 1.0, estCov)

    KalmanState(
      state.stateKey,
      state.stateIndex,
      estMean,
      estCov,
      residual,
      noiseCov)
  }

  def update(
    state: KalmanState,
    process: KalmanInput): KalmanState = {
    estimate(predict(state, process), process)
  }
}
