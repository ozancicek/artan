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

package com.ozancicek.artan.ml.linalg

import org.apache.spark.ml.linalg._
import org.scalatest.{FunSpec, Matchers}
import scala.math.abs


class LinalgUtilsSpec extends FunSpec with Matchers {

  def sumAbsoluteError(left: DenseMatrix, right: DenseMatrix): Double = {
    left.values.zip(right.values).foldLeft(0.0) { case(s, (l, r)) => s + abs(l-r)}
  }

  describe("LinalgUtils tests") {

    it("should calculate pinv") {
      val testMat = new DenseMatrix(2, 2, Array(4.0, 2.0, 2.0, 4.0))
      val inverseTest = LinalgUtils.pinv(testMat)
      val expectedEye = inverseTest.multiply(testMat)
      assert(sumAbsoluteError(expectedEye, DenseMatrix.eye(2)) < 10E-8)
    }

    it("should calculate sqrt") {
      val testMat = new DenseMatrix(2, 2, Array(4.0, 2.0, 2.0, 4.0))
      val sqrtTest = LinalgUtils.sqrt(testMat)
      val expectedTestMat = sqrtTest.multiply(sqrtTest)
      assert(sumAbsoluteError(expectedTestMat, testMat) < 10E-8)
    }
  }
}
