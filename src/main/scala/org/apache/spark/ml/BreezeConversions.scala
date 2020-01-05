package org.apache.spark.ml

import breeze.linalg.{Vector=>BV, DenseVector=>BDV, SparseVector=>BSV}
import breeze.linalg.{Matrix=>BM, DenseMatrix=>BDM, CSCMatrix=>BSM}

import org.apache.spark.ml.linalg.{Vectors, Matrices}
import org.apache.spark.ml.linalg.{Vector=>SV, DenseVector=>SDV, SparseVector=>SSV}
import org.apache.spark.ml.linalg.{Matrix=>SM, DenseMatrix=>SDM, SparseMatrix=>SSM}


object BreezeConversions {

  implicit class SparkToBreezeVector(vec: SV) {
    def asBreeze = vec.asBreeze
  }

  implicit class SparkToBreezeDenseVector(vec: SDV) {
    def asBreeze = vec.asBreeze
      .asInstanceOf[BDV[Double]]
  }

  implicit class BreezeToSparkVector(vec: BV[Double]) {
    def asSpark = Vectors.fromBreeze(vec)
  }

  implicit class BreezeToSparkDenseVector(vec: BDV[Double]) {
    def asSpark = Vectors.fromBreeze(vec)
      .asInstanceOf[SDV]
  }

  implicit class SparkToBreezeMatrix(mat: SM) {
    def asBreeze = mat.asBreeze
  }

  implicit class SparkToBreezeDenseMatrix(mat: SDM) {
    def asBreeze = mat.asBreeze
      .asInstanceOf[BDM[Double]]
  }

  implicit class BreezeToSparkMatrix(mat: BM[Double]) {
    def asSpark = Matrices.fromBreeze(mat)
  }

  implicit class BreezeToSparkDenseMatrix(mat: BDM[Double]) {
    def asSpark = Matrices.fromBreeze(mat)
      .asInstanceOf[SDM]
  }
}
