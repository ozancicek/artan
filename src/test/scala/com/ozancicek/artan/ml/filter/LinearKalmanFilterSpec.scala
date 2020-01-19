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

import breeze.stats.distributions.RandBasis
import com.ozancicek.artan.ml.testutils.StructuredStreamingTestWrapper
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.LAPACK
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.scalatest.{FunSpec, Matchers}
import java.sql.Timestamp
import scala.util.Random


case class LocalLinearMeasurement(measurement: DenseVector, eventTime: Timestamp)

case class LinearRegressionMeasurement(measurement: DenseVector, measurementModel: DenseMatrix)

case class DynamicLinearModel(measurement: DenseVector, processModel: DenseMatrix)


class LinearKalmanFilterSpec
  extends FunSpec
  with Matchers
  with StructuredStreamingTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)
  Random.setSeed(0)

  describe("Linear kalman filter tests") {
    describe("local linear trend") {

      val n = 100
      val dist = breeze.stats.distributions.Gaussian(0, 20)

      val ts = (0 until n).map(_.toDouble).toArray
      val zs = ts.map(t =>  t + dist.draw())
      val startTime = Timestamp.valueOf("2010-01-01 00:00:00.000")
      val timeDeltaSecs = 60L
      val measurements = zs.zip(ts).toSeq.map { case (z, t) =>
        val newTs = new Timestamp(startTime.getTime + t.toLong * timeDeltaSecs * 1000)
        LocalLinearMeasurement(new DenseVector(Array(z)), newTs)
      }

      val filter = new LinearKalmanFilter(2, 1)
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
        .setEventTimeCol("eventTime")

      val query = (in: Dataset[LocalLinearMeasurement]) => filter.transform(in)

      it("should have same result for batch & stream mode") {
        testAppendQueryAgainstBatch(measurements, query, "localLinearTrend")
      }

      it("should sort by event time") {
        val ts = query(Random.shuffle(measurements).toDS)
          .collect.map(_.getAs[Timestamp]("eventTime")).toList
        val sortedTs = ts.sortWith(_.compareTo(_) < 1)
        assert(ts == sortedTs)
      }

      it("should obtain the trend") {
        val batchState = query(Random.shuffle(measurements).toDS)
        val stats = batchState.groupBy($"stateKey")
          .agg(
            avg($"mahalanobis").alias("mahalanobis"),
            Summarizer.mean($"state").alias("avg"))
          .head

        assert(stats.getAs[Double]("mahalanobis") < 6.0)
        assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - n / 2) < 1.0)
        assert(scala.math.abs(stats.getAs[DenseVector]("avg")(1) - 1.0) < 1.0)
      }

    }

    describe("Ordinary least squares") {
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
      val measurements = zs.map { case (x, y, z) =>
        LinearRegressionMeasurement(
          new DenseVector(Array(z)),
          new DenseMatrix(1, 3, Array(x, y, 1)))
      }.toSeq

      val filter = new LinearKalmanFilter(3, 1)
        .setInitialCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))

      val query = (in: Dataset[LinearRegressionMeasurement]) => filter.transform(in)

      it("should have same solution with lapack dgels routine") {
        val modelState = query(measurements.toDS())
        val lastState = modelState.collect
          .filter(row=>row.getAs[Long]("stateIndex") == n)(0)
          .getAs[DenseVector]("state")

        // find least squares solution with dgels
        val features = new DenseMatrix(n, 3, xs ++ ys ++ Array.fill(n) {1.0})
        val target = new DenseVector(zs.map {case (x, y, z) => z})
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

      it("should have same result for batch & stream mode") {
        testAppendQueryAgainstBatch(measurements, query, "ols")
      }
    }

    describe("linear regression with time varying params") {
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
      }

      val measurements = (1 until n).map { i=>
        val dx = xs(i) - xs(i - 1)
        val measurement = new DenseVector(Array(zs(i)))
        val processModel = new DenseMatrix(
          2, 2, Array(1.0, 0.0, dx, 1.0))
        DynamicLinearModel(measurement, processModel)
      }

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

      val query = (in: Dataset[DynamicLinearModel]) => filter.transform(in)

      it("should filter the measurements") {
        val modelState = query(measurements.toDS())

        val lastState = modelState.collect
          .filter(row=>row.getAs[Long]("stateIndex") == n - 1)(0)
          .getAs[DenseVector]("state")

        val stats = modelState.groupBy($"stateKey")
          .agg(
            Summarizer.mean($"state").alias("avg"))
          .head
        assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - zs.reduce(_ + _)/zs.size) < 1.0)
      }

      it("should have same result for batch & stream mode") {
        testAppendQueryAgainstBatch(measurements, query, "dynamicLinearModel")
      }
    }
  }
}
