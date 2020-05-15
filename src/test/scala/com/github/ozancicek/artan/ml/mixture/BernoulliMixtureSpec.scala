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

import breeze.stats.distributions.{RandBasis, Bernoulli}
import org.apache.spark.sql.functions.max
import com.github.ozancicek.artan.ml.stats.{BernoulliDistribution, BernoulliMixtureDistribution}
import com.github.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.scalatest.{FunSpec, Matchers}


import scala.util.Random
import scala.math.abs

case class BernoulliSeq(sample: Boolean)

class BernoulliMixtureSpec
  extends FunSpec
    with Matchers
    with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)
  Random.setSeed(0L)

  def generateBernoulliSequence(
    size: Int,
    weights: Array[Double],
    dists: Array[BernoulliDistribution]): Seq[Boolean] = {

    val breezeDists = dists
      .map(s => new Bernoulli(s.probability))

    val measurements = breezeDists.zip(weights).flatMap {
      case (dist, w) => (0 until (w*size).toInt).map(i => dist.draw())
    }
    Random.shuffle(measurements.toSeq)
  }


  describe("Bernoulli mixture tests") {
    describe("test with three component bernoulli seq") {

      val size = 10000
      val weights = Array(0.4, 0.6)

      val g1 = BernoulliDistribution(0.7)
      val g2 = BernoulliDistribution(0.9)

      val dists = Array(g1, g2)

      val measurements = generateBernoulliSequence(size, weights, dists)
        .map(BernoulliSeq(_))

      val em = new BernoulliMixture(2)
        .setInitialProbabilities(Array(0.4, 0.8))
        .setStepSize(0.1)
        .setMinibatchSize(30)
        .setUpdateHoldout(1)

      val state = em.transform(measurements.toDF).cache()
      val maxSize = state.select(max("stateIndex")).head.getAs[Long](0)
      val lastState = state
        .filter(s"stateIndex = ${maxSize}")
        .select("mixtureModel.*").as[BernoulliMixtureDistribution].head()

      it("should find the clusters") {

        val probs = lastState.distributions.map(_.probability)
        val coeffs = lastState.weights

        val maeCoeffs = weights.indices.foldLeft(0.0) {
          case(s, i) => s + abs(weights(i) - coeffs(i))
        } / coeffs.length
        val coeffThreshold = 0.1

        val maeProbs = probs.indices.foldLeft(0.0) {
          case(s, i) => s + (probs(i) - dists(i).probability)
        } / coeffs.length
        val meanThreshold = 0.1

        assert(maeCoeffs < coeffThreshold)
        assert(maeProbs < meanThreshold)
      }
    }
  }
}
