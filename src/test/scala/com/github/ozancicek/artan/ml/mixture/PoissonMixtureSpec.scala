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
import com.github.ozancicek.artan.ml.stats.PoissonMixtureDistribution
import com.github.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.apache.spark.sql.functions.max
import org.scalatest.{FunSpec, Matchers}

import scala.util.Random

case class PoissonSeq(sample: Long)

class PoissonMixtureSpec
  extends FunSpec
    with Matchers
    with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)
  Random.setSeed(0L)

  def generatePoissonSequence(rates: List[Double], sizes: List[Int]): List[Int] = {
    val dists = rates.map(breeze.stats.distributions.Poisson(_))
    val seq = dists.zip(sizes).flatMap {
      case (dist, s) => (0 until s).map(i => dist.draw())
    }
    Random.shuffle(seq)
  }

  describe("Poisson mixture tests") {
    describe("test with three component poisson seq") {

      val inputRates = List(5.0, 10.0, 30.0)
      val size  = List(100, 700, 200)

      val counts = generatePoissonSequence(inputRates, size).map(PoissonSeq(_))

      val em = new PoissonMixture(3)
        .setInitialRates(Array(1.0, 7.0, 10.0))
        .setStepSize(0.1)
        .setMinibatchSize(5)

      val state = em.transform(counts.toDF)

      val maxSize = state.select(max("stateIndex")).head.getAs[Long](0)
      val lastState = state
        .filter(s"stateIndex = ${maxSize}")
        .select("mixtureModel.*").as[PoissonMixtureDistribution].head()

      it("should find the clusters") {

        val rates = lastState.distributions.map(_.rate)
        val coeffs = lastState.weights

        val expectedCoeffs = size.map(i=> i.toDouble/size.sum).toArray
        val expectedRates = inputRates.toArray

        val maeCoeffs = expectedCoeffs.indices.foldLeft(0.0) {
          case(s, i) => s + scala.math.abs(expectedCoeffs(i) - coeffs(i))
        } / coeffs.length
        val coeffThreshold = 0.1

        val maeRates = rates.indices.foldLeft(0.0) {
          case(s, i) => s + scala.math.abs(expectedRates(i) - rates(i))
        } / coeffs.length
        val rateThreshold = 1.0
        assert(maeCoeffs < coeffThreshold)
        assert(maeRates < rateThreshold)
      }
    }
  }
}
