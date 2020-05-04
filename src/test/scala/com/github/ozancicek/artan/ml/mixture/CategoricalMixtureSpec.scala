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

package com.github.ozancicek.artan.ml.mixture

import breeze.stats.distributions.RandBasis
import org.apache.spark.sql.functions.max
import com.github.ozancicek.artan.ml.stats.{CategoricalDistribution, CategoricalMixtureDistribution}
import com.github.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.scalatest.{FunSpec, Matchers}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.BreezeConversions._

import scala.util.Random
import scala.math.abs

case class CategoricalSeq(sample: Int)

class CategoricalMixtureSpec
  extends FunSpec
    with Matchers
    with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)
  Random.setSeed(0L)

  def generateCategoricalSequence(
    size: Int,
    weights: Array[Double],
    dists: Array[CategoricalDistribution]): Seq[Int] = {

    val breezeDists = dists
      .map(s => breeze.stats.distributions.Multinomial(s.probabilities.asBreeze))

    val measurements = breezeDists.zip(weights).flatMap {
      case (dist, w) => (0 until (w*size).toInt).map(i => dist.draw())
    }
    Random.shuffle(measurements.toSeq)
  }


  describe("Categorical mixture tests") {
    describe("test with three component categorical seq") {

      val size = 1000
      val weights = Array(0.55, 0.45)

      val g1 = CategoricalDistribution(
        new DenseVector(Array(0.6, 0.2, 0.15, 0.05)))
      val g2 = CategoricalDistribution(
        new DenseVector(Array(0.05, 0.15, 0.2, 0.6)))

      val dists = Array(g1, g2)

      val measurements = generateCategoricalSequence(size, weights, dists)
        .map(CategoricalSeq(_))

      val em = new CategoricalMixture(2)
        .setInitialProbabilities(Array(Array(0.26, 0.25, 0.25, 0.24), Array(0.24, 0.25, 0.25, 0.26)))
        .setStepSize(0.1)
        .setMinibatchSize(10)
        .setUpdateHoldout(1)

      val state = em.transform(measurements.toDF).cache()

      val maxSize = state.select(max("stateIndex")).head.getAs[Long](0)
      val lastState = state
        .filter(s"stateIndex = ${maxSize}")
        .select("mixtureModel.*").as[CategoricalMixtureDistribution].head()

      it("should find the clusters") {

        val probs = lastState.distributions.map(_.probabilities)
        val coeffs = lastState.weights

        val maeCoeffs = weights.indices.foldLeft(0.0) {
          case(s, i) => s + abs(weights(i) - coeffs(i))
        } / coeffs.length
        val coeffThreshold = 0.1

        val maeProbs = probs.indices.foldLeft(0.0) {
          case(s, i) => s + (probs(i).asBreeze - dists(i).probabilities.asBreeze).reduce(abs(_) + abs(_))
        } / (coeffs.length * 4)
        val meanThreshold = 0.2

        assert(maeCoeffs < coeffThreshold)
        assert(maeProbs < meanThreshold)
      }
    }
  }
}
