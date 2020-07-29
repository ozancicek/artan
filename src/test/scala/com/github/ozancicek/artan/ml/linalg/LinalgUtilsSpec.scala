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

package com.github.ozancicek.artan.ml.linalg

import com.github.ozancicek.artan.ml.testutils.SparkSessionTestWrapper
import org.apache.spark.ml.linalg._
import org.scalatest.{FunSpec, Matchers}
import scala.math.abs
import org.apache.spark.sql.functions._


class LinalgUtilsSpec extends FunSpec with Matchers with SparkSessionTestWrapper {
  import spark.implicits._

  def sumAbsoluteError(left: DenseMatrix, right: DenseMatrix): Double = {
    left.values.zip(right.values).foldLeft(0.0) { case(s, (l, r)) => s + abs(l-r)}
  }

  describe("LinalgUtils tests") {

    it("should calculate pinv") {
      val testMat = new DenseMatrix(2, 2, Array(4.0, 2.0, 2.0, 4.0))
      val inverseTest = LinalgUtils.pinv(testMat)(LinalgOptions(svdMethod = "dgesvd"))
      val expectedEye = inverseTest.multiply(testMat)
      assert(sumAbsoluteError(expectedEye, DenseMatrix.eye(2)) < 10E-8)
    }

    it("should calculate sqrt") {
      val testMat = new DenseMatrix(2, 2, Array(4.0, 2.0, 2.0, 4.0))
      val sqrtTest = LinalgUtils.sqrt(testMat)(LinalgOptions(svdMethod = "dgesdd"))
      val expectedTestMat = sqrtTest.multiply(sqrtTest)
      assert(sumAbsoluteError(expectedTestMat, testMat) < 10E-8)
    }

    it("should aggregate matrices") {
      val df = Seq(
        (1.0, DenseMatrix.eye(2)),
        (2.0, DenseMatrix.eye(2)),
        (3.0, DenseMatrix.eye(2)),
        (1.0, DenseMatrix.zeros(2, 2))).toDF("alpha", "mat")

      val aggFunction = LinalgUtils.axpyMatrixAggregate(2, 2)

      val agg = df.groupBy(lit(1))
        .agg(aggFunction($"alpha", $"mat").alias("agg"))
        .head.getAs[DenseMatrix](1)

      assert(agg == new DenseMatrix(2, 2, Array(6.0, 0.0, 0.0, 6.0)))
    }

    it("should aggregate latest state likelihood") {
      val df = Seq(
        (5.0, 1, new DenseVector(Array(1.0)), new DenseMatrix(1, 1, Array(1.0))),
        (5.0, 2, new DenseVector(Array(2.0)), new DenseMatrix(1, 1, Array(2.0))),
        (5.0, 3, new DenseVector(Array(3.0)), new DenseMatrix(1, 1, Array(3.0))),
        (5.0, 4, new DenseVector(Array(4.0)), new DenseMatrix(1, 1, Array(4.0))))
        .toDF("loglikelihood", "stateIndex", "state", "stateCovariance")

      val aggFunc = LinalgUtils.latestStateLikelihood
      val head = df.groupBy(lit(1))
        .agg(aggFunc($"loglikelihood", $"stateIndex", $"state", $"stateCovariance").alias("s"))
        .select("s.sumLoglikelihood", "s.stateIndex", "s.state", "s.stateCovariance")
        .head

      val sumll = head.getAs[Double](0)
      val index = head.getAs[Long](1)
      val state = head.getAs[DenseVector](2)
      val cov = head.getAs[DenseMatrix](3)

      assert(sumll == 20.0)
      assert(index == 4)
      assert(state == new DenseVector(Array(4.0)))
      assert(cov == new DenseMatrix(1, 1, Array(4.0)))
    }
  }
}
