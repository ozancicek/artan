package com.ozancicek.artan.ml.state

import org.apache.spark.ml.linalg.{Vector, Matrix, DenseVector}
import com.ozancicek.artan.ml.linalg.LinalgUtils
import com.ozancicek.artan.ml.stats.MultivariateGaussian


private[state] trait KeyedState {
  val groupKey: String
  val index: Long
}


private[ml] case class KalmanUpdate(
    groupKey: String,
    measurement: Option[Vector],
    measurementModel: Option[Matrix],
    measurementNoise: Option[Matrix],
    processModel: Option[Matrix],
    processNoise: Option[Matrix],
    control: Option[Vector],
    controlFunction: Option[Matrix])


case class KalmanState(
    groupKey: String,
    index: Long,
    mean: Vector,
    covariance: Matrix,
    residual: Vector,
    residualCovariance: Matrix) extends KeyedState {

  def loglikelihood: Double = {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    MultivariateGaussian.logpdf(residual.toDense, zeroMean, residualCovariance.toDense)
  }

  def mahalanobis: Double = {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    LinalgUtils.mahalanobis(residual.toDense, zeroMean, residualCovariance.toDense)
  }
}


case class RLSState(
    groupKey: String,
    index: Long,
    mean: Vector,
    covariance: Matrix) extends KeyedState


case class LMSState(
    groupKey: String,
    index: Long,
    mean: Vector) extends KeyedState


case class LMSUpdate(
    groupKey: String,
    label: Double,
    features: Vector)


case class RLSUpdate(
    groupKey: String,
    label: Double,
    features: Vector)
