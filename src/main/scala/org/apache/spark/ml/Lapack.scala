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

import org.apache.spark.ml.linalg._
import com.github.fommil.netlib.{LAPACK => NetlibLAPACK}
import com.github.fommil.netlib.LAPACK.getInstance
import org.netlib.util.intW

object LAPACK {

  @transient private var _lapack: NetlibLAPACK = _

  private def lapack: NetlibLAPACK = {
    if (_lapack == null) {
      _lapack = getInstance
    }
    _lapack
  }

  def upperTriangle(a: DenseMatrix): Unit = {
    for {
      i <- 0 until a.numCols
      j <- 0 until a.numRows
      if j > i
    } { a.values(j + i * a.numCols) = 0.0 }
  }

  def lowerTriangle(a: DenseMatrix): Unit = {
    for {
      i <- 0 until a.numCols
      j <- 0 until a.numRows
      if j < i
    } { a.values(j + i * a.numCols) = 0.0 }
  }

  /* LU solve b = a \ b */
  def dgesv(a: DenseMatrix, b: DenseMatrix): Unit = {
    val lda = if (!a.isTransposed) a.numRows else a.numCols
    val ldb = if (!b.isTransposed) b.numRows else b.numCols

    val info = new intW(0)
    lapack.dgesv(
      a.numRows,
      b.numCols,
      a.values,
      lda,
      new Array[Int](a.numRows),
      b.values,
      ldb,
      info
    )
    assert(info.`val` >= 0)
    if (info.`val` > 0) {
      throw new Exception("Not converged")
    }
  }

  /* cholesky decomposition of matrix a*/
  def dpotrf(a: DenseMatrix): Unit = {
    lowerTriangle(a)
    val lda = if (!a.isTransposed) a.numRows else a.numCols
    val info = new intW(0)
    lapack.dpotrf(
      "L",
      a.numRows,
      a.values,
      scala.math.max(1, a.numRows),
      info)
    assert(info.`val` >= 0)
    if (info.`val` > 0) {
      throw new Exception("Not converged")
    }
  }

  /* least squares*/
  def dgels(a: DenseMatrix, b: DenseVector): DenseVector = {
    val work = new Array[Double](math.max(1, a.numRows * a.numCols * 2))
    val mode = "N"
    val lda = if (!a.isTransposed) a.numRows else a.numCols
    val ldb = b.size
    val info = new intW(0)
    lapack.dgels(
      mode,
      a.numRows,
      a.numCols,
      1,
      a.values,
      lda,
      b.values,
      ldb,
      work,
      work.size,
      info)
    if (info.`val` < 0) {
      throw new Exception("Not converged")
    }
    new DenseVector(b.values.slice(0, a.numCols))
  }

  /* a = u * s * v**T */
  def dgesdd(
    a: DenseMatrix,
    u: DenseMatrix,
    s: DenseVector,
    v: DenseMatrix): Unit = {
    val mode = "A"
    val m = a.numRows
    val n = a.numCols
    val lda = if (!a.isTransposed) m else n
    val ldu = if (!u.isTransposed) u.numRows else u.numCols
    val ldv = if (!v.isTransposed) v.numRows else v.numCols

    val iwork = new Array[Int](8 * (m.min(n)))
    val workSize = (
      3L * scala.math.min(m, n) * scala.math.min(m, n) + scala.math.max(
        scala.math.max(m, n),
        4L * scala.math.min(m, n) * scala.math.min(m, n) + 4L * scala.math.min(m, n)))
    if (workSize >= Int.MaxValue) {
      throw new RuntimeException("Too large dimensions")
    }
    val work = new Array[Double](workSize.toInt)
    val info = new intW(0)

    lapack.dgesdd(
      mode,
      a.numRows,
      a.numCols,
      a.values,
      lda,
      s.values,
      u.values,
      ldu,
      v.values,
      ldv,
      work,
      work.length,
      iwork,
      info
    )
    if (info.`val` > 0) {
      throw new Exception("Not converged")
    }
    else if (info.`val` < 0) {
      throw new Exception("Invalid arguments")
    }
  }

}
