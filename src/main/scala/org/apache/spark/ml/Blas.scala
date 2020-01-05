package org.apache.spark.ml

import org.apache.spark.ml.linalg.{BLAS => SparkBLAS}
import com.github.fommil.netlib.{BLAS => NetlibBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import org.apache.spark.ml.linalg._


object BLAS {

  @transient private var _nativeBLAS: NetlibBLAS = _

  private def nativeBLAS: NetlibBLAS = {
    if (_nativeBLAS == null) {
      _nativeBLAS = NativeBLAS
    }
    _nativeBLAS
  }

  def axpy(a: Double, x: Vector, y: Vector): Unit = SparkBLAS.axpy(a, x, y)

  def axpy(a: Double, x: DenseMatrix, y: DenseMatrix): Unit = SparkBLAS.axpy(a, x, y)

  def scal(a: Double, x: Vector): Unit = SparkBLAS.scal(a, x)

  def dot(x: Vector, y: Vector): Double = SparkBLAS.dot(x, y)

  /* c := alpha * a * b + beta * c  */
  def gemm(alpha: Double,
           a: Matrix,
           b: DenseMatrix,
           beta: Double,
           c: DenseMatrix): Unit = SparkBLAS.gemm(alpha, a, b, beta, c)

  /* c := alpha * a * b + beta * c  */
  def gemv(alpha: Double,
           a: Matrix,
           b: Vector,
           beta: Double,
           c: DenseVector): Unit = SparkBLAS.gemv(alpha, a, b, beta, c)

  /* a := alpha * x * y.t + a */
  def dger(alpha: Double,
           x: DenseVector,
           y: DenseVector,
           a: DenseMatrix): Unit = {
    val lda = if (!a.isTransposed) a.numRows else a.numCols
    nativeBLAS.dger(
        a.numRows,
        a.numCols,
        alpha,
        x.values,
        1,
        y.values,
        1,
        a.values,
        lda
      )
  }

}
