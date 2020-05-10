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

package com.github.ozancicek.artan.ml
import org.apache.spark.sql.functions.{array, col, lit, randn, udf}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrices, Matrix, Vector, Vectors}
import org.apache.spark.ml.{BLAS, LAPACK}
import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction


object SparkFunctions {

  private def cholesky(in: Matrix): Matrix = {
    val result = in.toDense.copy
    LAPACK.dpotrf(result)
    result
  }

  private def randnVector(size: Int, seed: Long): Column = {
    arrayToVector(array((0 until size).map(i => randn(seed + i)): _*))
  }

  private def choleskyUDF: UserDefinedFunction = udf((in: Matrix) => cholesky(in))


  /**
   * UDF converting array of doubles to vector
   *
   * @return UserDefinedFunction
   */
  def arrayToVector: UserDefinedFunction = udf { arr: Seq[Double] =>
    Option(arr).map(in => Vectors.dense(in.toArray))
  }

  /**
   * UDF converting vector to array
   *
   * @return UserDefinedFunction
   */
  def vectorToArray: UserDefinedFunction = udf { vec: Vector =>
    Option(vec).map(in => in.toArray.toSeq)
  }

  /**
   * UDF for creating DenseMatrix from column major array of doubles
   *
   * @return UserDefinedFunction
   */
  def arrayToMatrix: UserDefinedFunction = udf { (numRows: Int, numCols: Int, arr: Seq[Double]) =>
    new DenseMatrix(numRows, numCols, arr.toArray)
  }

  /**
   * UDF for creating vector of zeros
   *
   * @return UserDefinedFunction
   */
  def zerosVector: UserDefinedFunction = udf { size: Int => Vectors.zeros(size)}

  /**
   * UDF for creating vector of ones
   *
   * @return UserDefinedFunction
   */
  def onesVector: UserDefinedFunction = udf { size: Int => new DenseVector(Array.fill(size) {1.0})}

  /**
   * UDF for dot product of two vectors
   *
   * @return UserDefinedFunction
   */
  def dotVector: UserDefinedFunction = udf { (x: Vector, y: Vector) =>
    (Option(x), Option(y)) match {
      case (Some(_), Some(_)) => Some(BLAS.dot(x, y))
      case _ => None
    }
  }

  /**
   * UDF for scaling vector with a scalar constant
   *
   * @return UserDefinedFunction
   */
  def scalVector: UserDefinedFunction = udf { (alpha: Double, x: Vector) =>
    (Option(alpha), Option(x)) match {
      case (Some(_), Some(_)) => {
        val result = x.copy
        BLAS.scal(alpha, result)
        Some(result)
      }
      case _ => None
    }
  }

  /**
   * UDF for axpy operation on vectors, alpha*x + y
   *
   * @return UserDefinedFunction
   */
  def axpyVector: UserDefinedFunction = udf { (alpha: Double, x: Vector, y: Vector) =>
    (Option(alpha), Option(x), Option(y)) match {
      case (Some(_), Some(_), Some(_)) => {
        val result = y.copy
        BLAS.axpy(alpha, x, result)
        Some(result)
      }
      case _ => None
    }
  }

  /**
   * UDF for creating identity matrix
   *
   * @return UserDefinedFunction
   */
  def eyeMatrix: UserDefinedFunction = udf { size: Int => Matrices.eye(size)}

  /**
   * UDF for creating zeros matrix
   *
   * @return UserDefinedFunction
   */
  def zerosMatrix: UserDefinedFunction = udf { (numRows: Int, numCols: Int) => Matrices.ones(numRows, numCols)}

  /**
   * UDF for creating diagonal matrix from vector
   *
   * @return UserDefinedFunction
   */
  def diagMatrix: UserDefinedFunction = udf { diag: Vector => Matrices.diag(diag)}

  /**
   * UDF for outer product of two vectors, a*x*y.T
   *
   * @return UserDefinedFunction
   */
  def outerProduct: UserDefinedFunction = udf { (alpha: Double, x: Vector, y: Vector) =>
    val result = Matrices.zeros(x.size, y.size).toDense
    BLAS.dger(alpha, x.toDense, y.toDense, result)
  }

  /**
   * UDF for scaling vector of unit normal samples to multivariate gaussian with mean and covariance specified with its
   * root. root covariance should be lower triangle, and it can be obtained with cholesky decomposition
   * of full covariance matrix.
   *
   * @return UserDefinedFunction
   */
  def scaleToMultiGaussian: UserDefinedFunction = udf { (mean: Vector, covRoot: Matrix, normal: Vector) =>
    val scaled = mean.toDense.copy
    BLAS.gemv(1.0, covRoot, normal, 1.0, scaled)
    scaled
  }

  /**
   * Sample from multivariate gaussian with literal distribution parameters
   *
   * @param mean mean vector
   * @param covariance covariance matrix
   * @param seed seed for random number generation
   * @return Column
   */
  def randMultiGaussian(mean: Vector, covariance: Matrix, seed: Long = 0): Column = {
    val root = cholesky(covariance)
    scaleToMultiGaussian(lit(mean), lit(root), randnVector(mean.size, seed))
  }

  /**
   * Sample from multivariate gaussian with column distribution parameters
   *
   * @param meanCol mean vector column
   * @param covCol covariance matrix column
   * @param size size of the mean vector
   * @param seed seed for random number generation
   * @return Column
   */
  def randMultiGaussianWithCol(meanCol: String, covCol: String, size: Int, seed: Long = 0): Column = {
    scaleToMultiGaussian(col(meanCol), choleskyUDF(col(covCol)), randnVector(size, seed))
  }

}
