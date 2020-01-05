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
