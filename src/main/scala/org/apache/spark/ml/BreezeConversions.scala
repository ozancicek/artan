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

import breeze.linalg.{Vector=>BV, DenseVector=>BDV}
import breeze.linalg.{Matrix=>BM, DenseMatrix=>BDM}

import org.apache.spark.ml.linalg.{Vectors, Matrices}
import org.apache.spark.ml.linalg.{Vector=>SV, DenseVector=>SDV}
import org.apache.spark.ml.linalg.{Matrix=>SM, DenseMatrix=>SDM}


object BreezeConversions {

  implicit class SparkToBreezeVector(vec: SV) {
    def asBreeze: BV[Double] = vec.asBreeze
  }

  implicit class SparkToBreezeDenseVector(vec: SDV) {
    def asBreeze: BDV[Double] = vec.asBreeze
      .asInstanceOf[BDV[Double]]
  }

  implicit class BreezeToSparkVector(vec: BV[Double]) {
    def asSpark: SV = Vectors.fromBreeze(vec)
  }

  implicit class BreezeToSparkDenseVector(vec: BDV[Double]) {
    def asSpark: SDV = Vectors.fromBreeze(vec)
      .asInstanceOf[SDV]
  }

  implicit class SparkToBreezeMatrix(mat: SM) {
    def asBreeze: BM[Double] = mat.asBreeze
  }

  implicit class SparkToBreezeDenseMatrix(mat: SDM) {
    def asBreeze: BDM[Double] = mat.asBreeze
      .asInstanceOf[BDM[Double]]
  }

  implicit class BreezeToSparkMatrix(mat: BM[Double]) {
    def asSpark: SM = Matrices.fromBreeze(mat)
  }

  implicit class BreezeToSparkDenseMatrix(mat: BDM[Double]) {
    def asSpark: SDM = Matrices.fromBreeze(mat)
      .asInstanceOf[SDM]
  }
}
