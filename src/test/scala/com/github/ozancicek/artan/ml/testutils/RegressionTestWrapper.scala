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

package com.github.ozancicek.artan.ml.testutils

import org.scalatest.{FunSpec, Matchers}
import org.apache.spark.ml.linalg._
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.LAPACK
import org.apache.spark.sql.{Dataset}
import org.apache.spark.ml.Transformer

import scala.math.{sqrt}


case class RegressionMeasurement(stateKey: String, measurement: DenseVector, measurementModel: DenseMatrix)


trait RegressionTestWrapper
  extends FunSpec
  with Matchers
  with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)

  def firstCoeff: Double = 0.5

  def secondCoeff: Double = -0.7
  def constCoeff: Double = 2.0
  def numSamples: Int = 40

  def generateMeasurements(
    f: Double => Double,
    stateKey: String = "stateKey"): (Seq[RegressionMeasurement], DenseMatrix, DenseVector) = {
    // linear regression data
    // z = a*x + b*y + c + N(0, 1)

    val dist = breeze.stats.distributions.Gaussian(0, 1)
    val xs = (0 until numSamples).map(i=>i.toDouble).toArray
    val ys = (0 until numSamples).map(i=> sqrt(i.toDouble)).toArray
    val zs = xs.zip(ys).map {
      case(x,y)=> (x, y, f(firstCoeff*x + secondCoeff*y + constCoeff) + dist.draw())
    }
    val measurements = zs.map { case (x, y, z) =>
      RegressionMeasurement(stateKey, new DenseVector(Array(z)), new DenseMatrix(1, 3, Array(x, y, 1)))
    }.toSeq
    val features = new DenseMatrix(numSamples, 3, xs ++ ys ++ Array.fill(numSamples) {1.0})
    val target = new DenseVector(zs.map {case (x, y, z) => z})
    (measurements, features, target)
  }


  def testLeastSquaresSolutionEquivalent[T <: Transformer](filter: T, threshold: Double): Unit = {

    val (measurements, features, target) = generateMeasurements(m => m)
    val query = (in: Dataset[RegressionMeasurement]) => filter.transform(in)

    val modelState = query(measurements.toDS)
    val lastState = modelState.collect
      .filter(row => row.getAs[Long]("stateIndex") == numSamples)(0)
      .getAs[DenseVector]("state")

    val coeffs = LAPACK.dgels(features, target)
    val mae = (0 until coeffs.size).foldLeft(0.0) {
      case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
    } / coeffs.size

    assert(mae < threshold)
  }

  def testLeastSquaresBatchStreamEquivalent[T <: Transformer](filter: T, testName: String): Unit = {
    val (measurements, features, target) = generateMeasurements(m => m)
    val query = (in: Dataset[RegressionMeasurement]) => filter.transform(in)
    testAppendQueryAgainstBatch(measurements, query, testName)
  }


  def testLogRegressionEquivalent[T <: Transformer](filter: T, threshold: Double): Unit = {

    val (measurements, _, _) = generateMeasurements(scala.math.exp)

    val query = (in: Dataset[RegressionMeasurement]) => filter.transform(in)

    val modelState = query(measurements.toDS())

    val lastState = modelState.collect
      .filter(row=>row.getAs[Long]("stateIndex") == numSamples)(0)
      .getAs[DenseVector]("state")
    val coeffs = new DenseVector(Array(firstCoeff, secondCoeff, constCoeff))
    val mae = (0 until coeffs.size).foldLeft(0.0) {
      case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
    } / coeffs.size

    assert(mae < threshold)
  }

  def testLogRegressionBatchStreamEquivalent[T <: Transformer](filter: T, testName: String): Unit = {
    val (measurements, features, target) = generateMeasurements(scala.math.exp)
    val query = (in: Dataset[RegressionMeasurement]) => filter.transform(in)
    testAppendQueryAgainstBatch(measurements, query, testName)
  }
}
