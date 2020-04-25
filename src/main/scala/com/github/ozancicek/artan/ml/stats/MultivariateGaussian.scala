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
import org.apache.spark.ml.LAPACK
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, Matrix}

import scala.math.{Pi, log, exp}


case class MultivariateGaussianDistribution(mean: Vector, covariance: Matrix) {
  def pdf(point: DenseVector): Double = exp(MultivariateGaussian.logpdf(point, mean.toDense, covariance.toDense))
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
    cov: DenseMatrix): Double = {

    val root = cov.copy
    LAPACK.dpotrf(root)
    val det = LinalgUtils.diag(root).values
      .map(log).reduce(_ + _)

    unnormalizedlogpdf(point, mean, cov) - (mean.size / 2.0 * log(2 * Pi) + det)
  }
}
