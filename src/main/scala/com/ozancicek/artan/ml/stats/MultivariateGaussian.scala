package com.ozancicek.artan.ml.stats

import org.apache.spark.ml.{BLAS, LAPACK}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import com.ozancicek.artan.ml.linalg.LinalgUtils
import scala.math.{log, Pi}

object MultivariateGaussian {

  def unnormalizedlogpdf(
    point: DenseVector,
    mean: DenseVector,
    cov: DenseMatrix): Double = {
    val dSquare = LinalgUtils.squaredMahalanobis(point, mean, cov)
    val res = - dSquare/ 2.0
    res
  }

  def logpdf(
    point: DenseVector,
    mean: DenseVector,
    cov: DenseMatrix): Double = {

    val root = cov.copy
    LAPACK.dpotrf(root)
    val det = LinalgUtils.diag(root).values
      .map(log).reduce(_ + _)

    unnormalizedlogpdf(point, mean, cov) - (mean.size / 2.0 * log(2 * Pi) + det)
  } 
}
