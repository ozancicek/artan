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

package com.ozancicek.artan.ml.filter

import breeze.stats.distributions.{RandBasis}
import com.ozancicek.artan.ml.testutils.SparkSessionTestWrapper
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.{LAPACK}
import org.apache.spark.sql.functions._
import org.scalatest.{FunSpec, Matchers}


class LinearKalmanFilterSpec
  extends FunSpec
  with Matchers
  with SparkSessionTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)

  describe("Batch linear kalman filter tests") {
    it("should filter local linear trend") {
      val n = 100
      val dist = breeze.stats.distributions.Gaussian(0, 20)

      val ts = (0 until n).map(_.toDouble).toArray
      val zs = ts.map(t =>  t + dist.draw())
      val df = zs.toSeq.map(z => ("1", new DenseVector(Array(z))))
        .toDF("modelID", "measurement")

      val filter = new LinearKalmanFilter(2, 1)
        .setStateKeyCol("modelID")
        .setMeasurementCol("measurement")
        .setInitialCovariance(
          new DenseMatrix(2, 2, Array(1000, 0, 0, 1000)))
        .setProcessModel(
          new DenseMatrix(2, 2, Array(1, 0, 1, 1)))
        .setProcessNoise(
          new DenseMatrix(2, 2, Array(0.0001, 0.0, 0.0, 0.0001)))
        .setMeasurementNoise(
          new DenseMatrix(1, 1, Array(20)))
        .setMeasurementModel(
          new DenseMatrix(1, 2, Array(1, 0)))
        .setCalculateMahalanobis

      val modelState = filter.transform(df)
      val stats = modelState.groupBy($"stateKey")
        .agg(
          avg($"mahalanobis").alias("mahalanobis"),
          Summarizer.mean($"state").alias("avg"))
        .head

      assert(stats.getAs[Double]("mahalanobis") < 6.0)
      assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - n/2) < 1.0)
      assert(scala.math.abs(stats.getAs[DenseVector]("avg")(1) - 1.0) < 1.0)
    }

    it("should be equivalent to ols") {
      // Ols problem
      // z = a*x + b*y + c + N(0, R)
      val n = 40
      val dist = breeze.stats.distributions.Gaussian(0, 1)

      val a = 1.5
      val b = -2.7
      val c = 5.0
      val xs = (0 until n).map(_.toDouble).toArray
      val ys = (0 until n).map(i=> scala.math.sqrt(i.toDouble)).toArray
      val zs = xs.zip(ys).map {
        case(x,y)=> (x, y, a*x + b*y + c + dist.draw())
      }
      val df = zs.map {
        case (x, y, z) => (new DenseVector(Array(z)), new DenseMatrix(1, 3, Array(x, y, 1)))
      }.toSeq.toDF("measurement", "measurementModel")

      val filter = new LinearKalmanFilter(3, 1)
        .setInitialCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))

      val modelState = filter.transform(df)

      val lastState = modelState.collect
        .filter(row=>row.getAs[Long]("stateIndex") == n)(0)
        .getAs[DenseVector]("state")

      // find least squares solution with dgels
      val features = new DenseMatrix(n, 3, xs ++ ys ++ Array.fill(n) {1.0})
      val target = new DenseVector(zs.map {case (x, y, z) => z}.toArray)
      val coeffs = LAPACK.dgels(features, target)
      // Error is mean absolute difference of kalman and least squares solutions
      val mae = (0 until coeffs.size).foldLeft(0.0) {
        case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
      } / coeffs.size
      // Error should be smaller than a certain threshold. The threshold is
      // tuned to some arbitrary small value depending on noise, cov and true coefficients.
      val threshold = 1E-4

      assert(mae < threshold)
    }

    it("linear regression with time varying params") {
      // Linear regression where params perform a random walk
      // z = a*x + b + N(0, R)
      // [a, b] = [a, b] + N(0, Q)
      val n = 100
      val R = breeze.stats.distributions.Gaussian(0, 1)
      val Q1 = breeze.stats.distributions.Gaussian(0, 0.1)
      val Q2 = breeze.stats.distributions.Gaussian(0, 0.15)

      var a = 1.5
      var b = -2.7
      val xs = (0 until n).map(_.toDouble).toArray
      val zs = xs.map { x =>
        a += Q1.draw()
        b += Q2.draw()
        val z = a*x + b + R.draw()
        z
      }.toArray

      val df = (1 until n).map { i=>
        val dx = xs(i) - xs(i - 1)
        val measurement = new DenseVector(Array(zs(i)))
        val processModel = new DenseMatrix(
          2, 2, Array(1.0, 0.0, dx, 1.0))
        (measurement, processModel)
      }.toSeq.toDF( "measurement", "processModel")

      val filter = new LinearKalmanFilter(2, 1)
        .setInitialCovariance(
          new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setProcessModelCol("processModel")
        .setMeasurementModel(new DenseMatrix(1, 2, Array(1.0, 0.0)))
        .setProcessNoise(
          new DenseMatrix(2, 2, Array(0.1, 0.0, 0.0, 0.05)))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.01)))
        .setCalculateMahalanobis
        .setCalculateLoglikelihood

      val modelState = filter.transform(df).cache()

      val lastState = modelState.collect
        .filter(row=>row.getAs[Long]("stateIndex") == n - 1)(0)
        .getAs[DenseVector]("state")

      val stats = modelState.groupBy($"stateKey")
        .agg(
          Summarizer.mean($"state").alias("avg"))
        .head
      assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - zs.reduce(_ + _)/zs.size) < 1.0)
    }
  }
}
