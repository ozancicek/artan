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

package org.apache.spark.ml

import org.scalatest.FunSpec
import org.apache.spark.ml.linalg.{Vectors, Matrices}
import org.scalatest.Matchers._
import org.apache.spark.ml.BreezeConversions._
import breeze.linalg._

class BreezeConversionsSpec
  extends FunSpec {

  describe("sparkml-breeze conversions") {

    it("should convert dense vector") {
      val sv = Vectors.dense(1.0, 1.0, 1.0, 1.0)
      val bv = DenseVector.ones[Double](4)
      assert(sv.asBreeze == bv)
      assert(sv == sv.asBreeze.asSpark)
    }

    it("should convert sparse vector") {
      val sv = Vectors
        .sparse(4, Array(0, 1, 2, 3), Array(1.0, 1.0, 1.0, 1.0))

      val bv = SparseVector.zeros[Double](4)
      bv(0 to 3) := 1.0

      assert(sv.asBreeze == bv)
      assert(sv == sv.asBreeze.asSpark)
    }

    it("should convert dense matrix") {
      val sm = Matrices
        .dense(2, 2, Array(1.0, 1.0, 1.0, 1.0))

      val bm = DenseMatrix.ones[Double](2, 2)

      assert(sm.asBreeze == bm)
      assert(sm == sm.asBreeze.asSpark)
    }

  }
}
