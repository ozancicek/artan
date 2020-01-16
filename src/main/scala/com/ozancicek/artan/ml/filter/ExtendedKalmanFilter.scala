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
    ExtendedKalmanStateSpec,
    ExtendedKalmanFilter]
  with HasProcessFunction with HasProcessStateJacobian with HasProcessNoiseJacobian
  with HasMeasurementFunction with HasMeasurementStateJacobian with HasMeasurementNoiseJacobian {

  def this(
    measurementSize: Int,
    stateSize: Int) = {
    this(measurementSize, stateSize, Identifiable.randomUID("extendedKalmanFilter"))
  }

  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  def setProcessStateJacobian(value: (Vector, Matrix) => Matrix): this.type = set(processStateJacobian, value)

  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  def setMeasurementStateJacobian(value: (Vector, Matrix) => Matrix): this.type = set(measurementStateJacobian, value)

  def setProcessNoiseJacobian(value: (Vector, Matrix) => Matrix): this.type = set(processNoiseJacobian, value)

  def setMeasurementNoiseJacobian(value: (Vector, Matrix) => Matrix): this.type = set(measurementNoiseJacobian, value)


  override def copy(extra: ParamMap): ExtendedKalmanFilter = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = withExtraColumns(filter(dataset))

  protected def stateUpdateSpec: ExtendedKalmanStateSpec = new ExtendedKalmanStateSpec(
    getInitialState,
    getInitialCovariance,
    getFadingFactor,
    getProcessFunctionOpt,
    getProcessStateJacobianOpt,
    getProcessNoiseJacobianOpt,
    getMeasurementFunctionOpt,
    getMeasurementStateJacobianOpt,
    getMeasurementNoiseJacobianOpt
  )
}


private[filter] class ExtendedKalmanStateSpec(
    val stateMean: Vector,
    val stateCov: Matrix,
    val fadingFactor: Double,
    val processFunction: Option[(Vector, Matrix) => Vector],
    val processStateJacobian: Option[(Vector, Matrix) => Matrix],
    val processNoiseJacobian: Option[(Vector, Matrix) => Matrix],
    val measurementFunction: Option[(Vector, Matrix) => Vector],
    val measurementStateJacobian: Option[(Vector, Matrix) => Matrix],
    val measurementNoiseJacobian: Option[(Vector, Matrix) => Matrix])
  extends KalmanStateUpdateSpec[ExtendedKalmanStateCompute] {

  val kalmanCompute = new ExtendedKalmanStateCompute(
    fadingFactor,
    processFunction,
    processStateJacobian,
    processNoiseJacobian,
    measurementFunction,
    measurementStateJacobian,
    measurementNoiseJacobian)
}

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
