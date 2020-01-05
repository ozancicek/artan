package com.ozancicek.artan.ml.linalg

import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, SparseMatrix}
import org.apache.spark.ml.{BLAS, LAPACK}
import scala.math.{sqrt => scalarSqrt}


object LinalgUtils {

  def upperTriangle(a: DenseMatrix): Unit = {
    for {
      i <- 0 until a.numCols
      j <- 0 until a.numRows
      if(j > i)
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

  private def svdDiagOp(mat: DenseMatrix, diagOp: (Double) => Double): DenseMatrix = {
    val m = mat.numRows
    val n = mat.numCols
    val nSingular = if (m < n) m else n
    val s = new DenseVector(new Array[Double](nSingular))
    val u = DenseMatrix.zeros(m, m)
    val v = DenseMatrix.zeros(n, n)
    LAPACK.dgesdd(mat.copy, u, s, v)

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

  def pinv(mat: DenseMatrix) = svdDiagOp(mat, (v: Double) => if (v == 0.0) 0.0f else 1 / v)
  def sqrt(mat: DenseMatrix) = svdDiagOp(mat, scalarSqrt)

  def squaredMahalanobis(
    point: DenseVector,
    mean: DenseVector,
    cov: DenseMatrix): Double = {
    val centered = point.copy
    BLAS.axpy(-1.0, mean, centered)

    val slv = centered.copy
    LAPACK.dgesv(cov, new DenseMatrix(slv.size, 1, slv.values))
    BLAS.dot(slv, centered)
  }

  def mahalanobis(
    point: DenseVector,
    mean: DenseVector,
    cov: DenseMatrix): Double = scalarSqrt(squaredMahalanobis(point, mean, cov))
}
