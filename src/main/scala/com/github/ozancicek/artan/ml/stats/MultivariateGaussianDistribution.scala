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

package com.github.ozancicek.artan.ml.stats

import com.github.ozancicek.artan.ml.linalg.LinalgUtils
import org.apache.spark.ml.{BLAS, LAPACK}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, Vector}

import scala.math.{Pi, log}


case class MultivariateGaussianDistribution(mean: Vector, covariance: Matrix)
  extends Distribution[Vector, MultivariateGaussianDistribution] {

  override def loglikelihood(sample: Vector): Double = MultivariateGaussian
    .logpdf(sample.toDense, mean.toDense, covariance.toDense)

  override def summarize(weights: Seq[Double], samples: Seq[Vector]): MultivariateGaussianDistribution = {
    val meanSummary = new DenseVector(Array.fill(mean.size){0.0})
    val covSummary = DenseMatrix.zeros(covariance.numRows, covariance.numCols)
    samples.zip(weights).foreach { case(v, d) =>
      BLAS.axpy(d/samples.length, v, meanSummary)
      val residual = v.toDense.copy
      BLAS.axpy(-1.0, mean, residual)
      BLAS.dger(d/samples.length, residual, residual, covSummary)
    }
    MultivariateGaussianDistribution(meanSummary, covSummary)
  }

  override def scal(weight: Double): MultivariateGaussianDistribution = {
    val weightedMean = mean.toDense.copy
    BLAS.scal(weight, weightedMean)
    val weightedCov = DenseMatrix.zeros(mean.size, mean.size)
    BLAS.axpy(weight, covariance.toDense, weightedCov)
    MultivariateGaussianDistribution(weightedMean, weightedCov)
  }

  override def axpy(weight: Double, other: MultivariateGaussianDistribution): MultivariateGaussianDistribution = {
    val newMean = mean.copy
    BLAS.axpy(weight, other.mean, newMean)
    val newCov = covariance.toDense.copy
    BLAS.axpy(weight, other.covariance.toDense, newCov)
    MultivariateGaussianDistribution(newMean, newCov)
  }
}


private[ml] object MultivariateGaussian {

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
    cov: DenseMatrix,
    covRoot: Option[DenseMatrix] = None): Double = {

    val root = covRoot match {
      case Some(r) => r
      case None => {
        val r = cov.copy
        LAPACK.dpotrf(r)
        r
      }
    }

    val det = LinalgUtils.diag(root).values
      .map(log).reduce(_ + _)

    unnormalizedlogpdf(point, mean, cov) - (mean.size / 2.0 * log(2 * Pi) + det)
  }
}
