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

package com.github.ozancicek.artan.ml.em

import breeze.stats.distributions.RandBasis
import com.github.ozancicek.artan.ml.state.GaussianMixtureModel
import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution
import com.github.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.scalatest.{FunSpec, Matchers}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.BreezeConversions._

import scala.util.Random
import scala.math.abs

case class GaussianSeq(measurement: DenseVector)

class MultivariateGaussianMixtureSpec
  extends FunSpec
    with Matchers
    with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)
  Random.setSeed(0L)

  def generateGMSequence(
    size: Int,
    weights: Array[Double],
    gaussians: Array[MultivariateGaussianDistribution]): Seq[DenseVector] = {

    val breezeDists = gaussians
      .map(s => breeze.stats.distributions.MultivariateGaussian(s.mean.toDense.asBreeze, s.covariance.toDense.asBreeze))

    val measurements = breezeDists.zip(weights).flatMap {
      case (dist, w) => (0 until (w*size).toInt).map(i => dist.draw().asSpark)
    }
    Random.shuffle(measurements.toSeq)
  }

  describe("Multivariate gaussian mixture tests") {
    describe("test with three component gaussian seq") {

      val size = 5000
      val weights = Array(0.2, 0.3, 0.5)

      val g1 = MultivariateGaussianDistribution(
        new DenseVector(Array(10.0, 2.0)),
        new DenseMatrix(2, 2, Array(2.0, 1.0, 1.0, 2.0)))
      val g2 = MultivariateGaussianDistribution(
        new DenseVector(Array(1.0, 4.0)),
        new DenseMatrix(2, 2, Array(4.0, 0.0, 0.0, 4.0)))
      val g3 = MultivariateGaussianDistribution(
        new DenseVector(Array(5.0, 3.0)),
        new DenseMatrix(2, 2, Array(5.0, 3.0, 3.0, 5.0)))
      val gaussians = Array(g1, g2, g3)

      val measurements = generateGMSequence(size, weights, gaussians)
        .map(GaussianSeq(_))

      val eye = Array(1.0, 0.0, 0.0, 1.0)
      val em = new MultivariateGaussianMixture(3)
        .setMeasurementSize(2)
        .setInitialMeans(Array(Array(9.0, 9.0), Array(1.0, 1.0), Array(5.0, 5.0)))
        .setInitialCovariances(Array(eye, eye, eye))
        .setStepSize(0.01)

      val state = em.transform(measurements.toDF)

      val lastState = state
        .filter(s"stateIndex = ${size}")
        .select("mixtureModel.*").as[GaussianMixtureModel].head()

      it("should find the clusters") {

        val means = lastState.distributions.map(_.mean)
        val covs = lastState.distributions.map(_.covariance)
        val coeffs = lastState.weights

        val maeCoeffs = weights.indices.foldLeft(0.0) {
          case(s, i) => s + abs(weights(i) - coeffs(i))
        } / coeffs.length
        val coeffThreshold = 0.1

        val maeMeans = means.indices.foldLeft(0.0) {
          case(s, i) => s + (means(i).asBreeze - gaussians(i).mean.asBreeze).reduce(abs(_) + abs(_))
        } / (coeffs.length * 2)
        val meanThreshold = 0.5

        val maeCovs = covs.indices.foldLeft(0.0) {
          case(s, i) => s + (covs(i).asBreeze - gaussians(i).covariance.asBreeze)
            .valuesIterator.reduce(abs(_) + abs(_))
        } / (coeffs.length * 4)
        val covsThreshold = 2.0

        assert(maeCoeffs < coeffThreshold)
        assert(maeMeans < meanThreshold)
        assert(maeCovs < covsThreshold)
      }
    }
  }
}
