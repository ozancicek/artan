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

package com.github.ozancicek.artan.ml.filter

import breeze.stats.distributions.RandBasis
import com.github.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.LAPACK
import org.apache.spark.sql.Dataset
import org.scalatest.{FunSpec, Matchers}

case class RLSMeasurement(label: Double, features: DenseVector)

class RecursiveLeastSquaresFilterSpec
  extends FunSpec
  with Matchers
  with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)

  describe("Batch RLS tests") {
    describe("should be equivalent to ols") {
      // Ols problem
      // z = a*x + b*y + c + N(0, R)
      val n = 100
      val dist = breeze.stats.distributions.Gaussian(0, 1)

      val a = 1.5
      val b = -2.7
      val c = 5.0
      val xs = (0 until n).map(_.toDouble).toArray
      val ys = (0 until n).map(i=> scala.math.sqrt(i.toDouble)).toArray
      val zs = xs.zip(ys).map {
        case(x,y)=> (x, y, a*x + b*y + c + dist.draw())
      }
      val measurements = zs.map {
        case (x, y, z) => RLSMeasurement(z, new DenseVector(Array(x, y, 1)))
      }.toSeq

      // set large regularization factor to behave like ols
      val filter = new RecursiveLeastSquaresFilter(3)
        .setRegularizationMatrixFactor(10E5)

      val query = (in: Dataset[RLSMeasurement]) => filter.transform(in)

      it("should have same solution with lapack dgels routine") {
        val modelState = query(measurements.toDS)

        val lastState = modelState.collect
          .filter(row=>row.getAs[Long]("stateIndex") == n)(0)
          .getAs[DenseVector]("state")

        // find least squares solution with dgels
        val features = new DenseMatrix(n, 3, xs ++ ys ++ Array.fill(n) {1.0})
        val target = new DenseVector(zs.map {case (x, y, z) => z}.toArray)
        val coeffs = LAPACK.dgels(features, target)
        // Error is mean absolute difference of kalman and least squares solutions
        val mae = (0 until coeffs.size).foldLeft(0.0) {
          case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
        } / coeffs.size
        // Error should be smaller than a certain threshold. The threshold is
        // tuned to some arbitrary small value depending on noise, cov and true coefficients.
        val threshold = 1E-4

        assert(mae < threshold)
      }

      it("should have same result for batch & stream mode") {
        testAppendQueryAgainstBatch(measurements, query, "RLSModel")
      }
    }
  }
}
