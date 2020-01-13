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


class UnscentedKalmanFilter(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends KalmanTransformer[
    UnscentedKalmanStateCompute,
    UnscentedKalmanStateEstimator]
  with KalmanUpdateParams with HasInitialState with HasInitialCovariance with HasFadingFactor
  with HasProcessFunction with HasMeasurementFunction with SigmaPointsParams {

  def this(
    measurementSize: Int,
    stateSize: Int) = {
    this(measurementSize, stateSize, Identifiable.randomUID("unscentedKalmanFilter"))
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

  def setProcessFunction(value: (Vector, Matrix) => Vector): this.type = set(processFunction, value)

  def setMeasurementFunction(value: (Vector, Matrix) => Vector): this.type = set(measurementFunction, value)

  def setCalculateLoglikelihood: this.type = set(calculateLoglikelihood, true)

  def setCalculateMahalanobis: this.type = set(calculateMahalanobis, true)

  def setSigmaPoints(value: String): this.type = set(sigmaPoints, value)

  def setMerweAlpha(value: Double): this.type = set(merweAlpha, value)

  def setMerweBeta(value: Double): this.type = set(merweBeta, value)

  def setMerweKappa(value: Double): this.type = set(merweKappa, value)

  def setJulierKappa(value: Double): this.type = set(julierKappa, value)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = withExtraColumns(filter(dataset))

  protected def stateUpdateFunc: UnscentedKalmanStateEstimator = new UnscentedKalmanStateEstimator(
    getInitialState,
    getInitialCovariance,
    getFadingFactor,
    getSigmaPoints,
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
      state.stateKey,
      state.stateIndex + 1,
      stateMean,
      stateCov,
      state.residual,
      state.residualCovariance)
  }


  private def estimate(
    state: KalmanState,
    process: KalmanInput): KalmanState = {

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

    KalmanState(
      state.stateKey, state.stateIndex, newMean, newCov, residual, estimateCov)
  }

  def update(
    state: KalmanState,
    process: KalmanInput): KalmanState = {
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
