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

package com.ozancicek.artan.ml.state

import org.apache.spark.ml.linalg.{Vector, Matrix, DenseVector}
import com.ozancicek.artan.ml.linalg.LinalgUtils
import com.ozancicek.artan.ml.stats.MultivariateGaussian


private[ml] case class KalmanUpdate(
    groupKey: String,
    measurement: Option[Vector],
    measurementModel: Option[Matrix],
    measurementNoise: Option[Matrix],
    processModel: Option[Matrix],
    processNoise: Option[Matrix],
    control: Option[Vector],
    controlFunction: Option[Matrix])


private[ml] case class KalmanState(
    groupKey: String,
    index: Long,
    mean: Vector,
    covariance: Matrix,
    residual: Vector,
    residualCovariance: Matrix) extends KeyedState[String, KalmanOutput] {

  def asOut: KalmanOutput = {
    KalmanOutput(
      groupKey,
      index,
      mean,
      covariance,
      residual,
      residualCovariance
    )
  }
}


case class KalmanOutput(
    groupKey: String,
    index: Long,
    mean: Vector,
    covariance: Matrix,
    residual: Vector,
    residualCovariance: Matrix) {

  def loglikelihood: Double = {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    MultivariateGaussian.logpdf(residual.toDense, zeroMean, residualCovariance.toDense)
  }

  def mahalanobis: Double = {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    LinalgUtils.mahalanobis(residual.toDense, zeroMean, residualCovariance.toDense)
  }
}
