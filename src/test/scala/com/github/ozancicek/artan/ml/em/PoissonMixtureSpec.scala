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
import com.github.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.scalatest.{FunSpec, Matchers}
import org.apache.spark.ml.linalg._
import scala.util.Random

case class PoissonSeq(count: Long)

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
        .setInitialPoissonRates(new DenseVector(Array(1.0, 7.0, 10.0)))

      val state = em.transform(counts.toDF)

      val lastState = state.filter(s"stateIndex = ${size.sum}").head()

      it("should find the clusters") {
        val rates = lastState.getAs[DenseVector]("rates")
        val coeffs = lastState.getAs[DenseVector]("mixtureCoefficients")

        val expectedCoeffs = new DenseVector(size.map(i=> i.toDouble/size.sum).toArray)
        val expectedRates = new DenseVector(inputRates.toArray)

        val maeCoeffs = (0 until expectedCoeffs.size).foldLeft(0.0) {
          case(s, i) => s + scala.math.abs(expectedCoeffs(i) - coeffs(i))
        } / coeffs.size
        val coeffThreshold = 0.1

        val maeRates = (0 until rates.size).foldLeft(0.0) {
          case(s, i) => s + scala.math.abs(expectedRates(i) - rates(i))
        } / coeffs.size
        val rateThreshold = 1.0
        assert(maeCoeffs < coeffThreshold)
        assert(maeRates < rateThreshold)
      }
    }
  }
}
