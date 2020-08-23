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

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix, Vector}
import org.apache.spark.ml.{BLAS, LAPACK}
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.{Aggregator, UserDefinedFunction}
import org.apache.spark.sql.functions.udaf
import org.apache.spark.SparkConf

import scala.math.{sqrt => scalarSqrt}


private[artan] case class LinalgOptions(svdMethod: String = "dgesdd", raiseExceptions: Boolean = true)


private[artan] object LinalgOptions {

  def fromSparkConf(conf: SparkConf): LinalgOptions = {
    val svdMethod = conf.get("spark.artan.ml.linalg.svdMethod", "dgesdd")
    if (!Set("dgesdd", "dgesvd").contains(svdMethod)) {
      throw new Exception(s"svdMethod must be either dgesdd or dgesvd, provided: $svdMethod")
    }
    val raiseExceptions = conf.getBoolean("spark.ml.linalg.raiseExceptions", true)
    LinalgOptions(svdMethod, raiseExceptions)
  }
}


private[ml] object LinalgUtils {

  case class VectorAxpyInput(a: Double, x: Vector)

  class VectorAxpy(size: Int) extends Aggregator[VectorAxpyInput, Vector, Vector] with Serializable {

    private def zeroVector = new DenseVector(Array.fill(size) {0.0})

    override def zero: Vector = zeroVector

    override def reduce(b: Vector, a: VectorAxpyInput): Vector = {
      val y = b.copy
      BLAS.axpy(a.a, a.x, y)
      y
    }

    override def merge(b1: Vector, b2: Vector): Vector = {
      val y = b2.copy
      BLAS.axpy(1.0, b1, y)
      y
    }

    override def finish(reduction: Vector): Vector = reduction

    override def bufferEncoder: Encoder[Vector] = implicitly(ExpressionEncoder[Vector]())

    override def outputEncoder: Encoder[Vector] = implicitly(ExpressionEncoder[Vector]())

  }


  case class MatrixAxpyInput(a: Double, x: Matrix)

  private class MatrixAxpy(numRows: Int, numCols: Int) extends Aggregator[MatrixAxpyInput, Matrix, Matrix] {

    private def zeroMatrix = DenseMatrix.zeros(numRows, numCols)

    override def zero: Matrix = zeroMatrix

    override def reduce(b: Matrix, a: MatrixAxpyInput): Matrix = {
      val y = b.toDense.copy
      BLAS.axpy(a.a, a.x.toDense, y)
      y
    }

    override def merge(b1: Matrix, b2: Matrix): Matrix = {
      val y = b1.toDense.copy
      BLAS.axpy(1.0, b2.toDense, y)
      y
    }

    override def finish(reduction: Matrix): Matrix = reduction

    override def bufferEncoder: Encoder[Matrix] = implicitly(ExpressionEncoder[Matrix]())

    override def outputEncoder: Encoder[Matrix] = implicitly(ExpressionEncoder[Matrix]())

  }

  case class StateLikelihood(loglikelihood: Double, stateIndex: Long, state: Option[Vector], covariance: Option[Matrix])

  private class LatestStateLikelihood extends Aggregator[StateLikelihood, StateLikelihood, StateLikelihood] {

    override def zero: StateLikelihood = StateLikelihood(0.0, Long.MinValue, None, None)

    override def reduce(b: StateLikelihood, a: StateLikelihood): StateLikelihood = {
      val llSum = b.loglikelihood + a.loglikelihood
      if (a.stateIndex >= b.stateIndex) {
        StateLikelihood(llSum, a.stateIndex, a.state, a.covariance)
      } else {
        StateLikelihood(llSum, b.stateIndex, b.state, b.covariance)
      }
    }

    override def merge(b1: StateLikelihood, b2: StateLikelihood): StateLikelihood = reduce(b1, b2)

    override def finish(reduction: StateLikelihood): StateLikelihood = reduction

    override def bufferEncoder: Encoder[StateLikelihood] = implicitly(ExpressionEncoder[StateLikelihood]())

    override def outputEncoder: Encoder[StateLikelihood] = implicitly(ExpressionEncoder[StateLikelihood]())

  }

  def axpyMatrixAggregate(numRows: Int, numCols: Int): UserDefinedFunction = udaf(new MatrixAxpy(numRows, numCols))

  def axpyVectorAggregate(size: Int): UserDefinedFunction  = udaf(new VectorAxpy(size))

  def latestStateLikelihood: UserDefinedFunction = udaf(new LatestStateLikelihood)

  def upperTriangle(a: DenseMatrix): Unit = {
    for {
      i <- 0 until a.numCols
      j <- 0 until a.numRows
      if j > i
    } { a.values(j + i * a.numCols) = 0.0 }
  }

  def diag(a: DenseMatrix): DenseVector = {
    val values = (0 until a.numRows)
      .zipWithIndex
      .map(t=> t._1 + t._2 * a.numCols)
      .map(a.values(_))
      .toArray
    new DenseVector(values)
  }

  def lagOp(n: Int): SparseMatrix = {
    val rowIndices = 1 until n
    val colPtrs = 0 +: rowIndices :+ n - 1
    val values = Array.fill(n - 1) {0.0}
    new SparseMatrix(n, n, colPtrs.toArray, rowIndices.toArray, values)
  }

  private def svdDiagOp(mat: DenseMatrix, diagOp: Double => Double)(implicit ops: LinalgOptions): DenseMatrix = {
    val m = mat.numRows
    val n = mat.numCols
    val nSingular = if (m < n) m else n
    val s = new DenseVector(new Array[Double](nSingular))
    val u = DenseMatrix.zeros(m, m)
    val v = DenseMatrix.zeros(n, n)

    if (ops.svdMethod == "dgesvd") {
      LAPACK.dgesvd(mat.copy, u, s, v, ops.raiseExceptions)
    } else {
      LAPACK.dgesdd(mat.copy, u, s, v, ops.raiseExceptions)
    }

    val si = s.values.map(diagOp)

    val siDiagValues = new Array[Double](u.numCols * v.numRows)
    for {
      i <- 0 to u.numCols
      j <- 0 to v.numRows
      if( i == j && i < math.min(u.numCols, v.numRows))
    } { siDiagValues(j + i * u.numCols) = si(i) }

    val siDiag = new DenseMatrix(u.numCols, v.numRows, siDiagValues)

    val result = DenseMatrix.zeros(m, n)
    BLAS.gemm(1.0, u, siDiag, 0.0, result)
    BLAS.gemm(1.0, result.copy, v, 0.0, result)
    result.transpose
  }

  def pinv(mat: DenseMatrix)(implicit ops: LinalgOptions): DenseMatrix = {
    svdDiagOp(mat, (v: Double) => if (v == 0.0) 0.0f else 1 / v)
  }

  def sqrt(mat: DenseMatrix)(implicit ops: LinalgOptions): DenseMatrix = svdDiagOp(mat, scalarSqrt)

  def squaredMahalanobis(
    point: DenseVector,
    mean: DenseVector,
    cov: DenseMatrix): Double = {
    val centered = point.copy
    BLAS.axpy(-1.0, mean, centered)

    val slv = centered.copy
    LAPACK.dgesv(cov.copy, new DenseMatrix(slv.size, 1, slv.values))
    BLAS.dot(slv, centered)
  }

  def mahalanobis(
    point: DenseVector,
    mean: DenseVector,
    cov: DenseMatrix): Double = scalarSqrt(squaredMahalanobis(point, mean, cov))
}
