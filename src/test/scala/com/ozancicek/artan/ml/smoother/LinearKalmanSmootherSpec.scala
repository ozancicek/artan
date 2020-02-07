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

package com.ozancicek.artan.ml.smoother

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.scalatest.{FunSpec, Matchers}
import com.ozancicek.artan.ml.testutils.RegressionTestWrapper
import java.sql.Timestamp
import scala.util.Random


case class LocalLinearMeasurement(measurement: DenseVector, eventTime: Timestamp)

case class LinearRegressionMeasurement(measurement: DenseVector, measurementModel: DenseMatrix)

case class DynamicLinearModel(measurement: DenseVector, processModel: DenseMatrix)


class LinearKalmanSmootherSpec
  extends FunSpec
  with Matchers
  with RegressionTestWrapper {

  import spark.implicits._
  override def numSamples: Int = 100

  describe("Linear kalman smoother tests") {
    describe("local linear trend") {

      val dist = breeze.stats.distributions.Gaussian(0, 20)

      val ts = (0 until numSamples).map(_.toDouble).toArray
      val zs = ts.map(t =>  t + dist.draw())
      val startTime = Timestamp.valueOf("2010-01-01 00:00:00.000")
      val timeDeltaSecs = 60L * 10L
      val measurements = zs.zip(ts).toSeq.map { case (z, t) =>
        val newTs = new Timestamp(startTime.getTime + t.toLong * timeDeltaSecs * 1000)
        LocalLinearMeasurement(new DenseVector(Array(z)), newTs)
      }

      val filter = new LinearKalmanSmoother(2, 1)
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
        .setEventTimeCol("eventTime")
        .setFixedLag(numSamples)

      val query = (in: Dataset[LocalLinearMeasurement]) => filter.transform(in)

      it("should have same result for batch & stream mode") {
        testAppendQueryAgainstBatch(measurements, query, "localLinearTrendSmoothing")
      }

      it("should obtain the trend") {
        val batchState = query(Random.shuffle(measurements).toDS)
        val stats = batchState.groupBy($"stateKey")
          .agg(Summarizer.mean($"state").alias("avg"))
          .head
        assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - numSamples / 2) < 2.0)
        assert(scala.math.abs(stats.getAs[DenseVector]("avg")(1) - 1.0) < 1.0)
      }

    }


    describe("Ordinary least squares") {

      val filter = new LinearKalmanSmoother(3, 1)
        .setInitialCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))
        .setFixedLag(numSamples)


      it("should have same solution with lapack dgels routine") {
        testLeastSquaresSolutionEquivalent(filter, 10E-4)

      }

      it("should have same result for batch & stream mode") {
        testLeastSquaresBatchStreamEquivalent(filter, "LKSOls")
      }
    }

    describe("linear regression with time varying params") {
      // Linear regression where params perform a random walk
      // z = a*x + b + N(0, R)
      // [a, b] = [a, b] + N(0, Q)
      val R = breeze.stats.distributions.Gaussian(0, 1)
      val Q1 = breeze.stats.distributions.Gaussian(0, 0.1)
      val Q2 = breeze.stats.distributions.Gaussian(0, 0.15)

      var a = 1.5
      var b = -2.7
      val xs = (0 until numSamples).map(_.toDouble).toArray
      val zs = xs.map { x =>
        a += Q1.draw()
        b += Q2.draw()
        val z = a*x + b + R.draw()
        z
      }

      val measurements = (1 until numSamples).map { i=>
        val dx = xs(i) - xs(i - 1)
        val measurement = new DenseVector(Array(zs(i)))
        val processModel = new DenseMatrix(
          2, 2, Array(1.0, 0.0, dx, 1.0))
        DynamicLinearModel(measurement, processModel)
      }

      val filter = new LinearKalmanSmoother(2, 1)
        .setInitialCovariance(
          new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setProcessModelCol("processModel")
        .setMeasurementModel(new DenseMatrix(1, 2, Array(1.0, 0.0)))
        .setProcessNoise(
          new DenseMatrix(2, 2, Array(0.1, 0.0, 0.0, 0.05)))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.01)))
        .setFixedLag(numSamples - 1)

      val query = (in: Dataset[DynamicLinearModel]) => filter.transform(in)

      it("should filter the measurements") {
        val modelState = query(measurements.toDS())

        val lastState = modelState.collect
          .filter(row=>row.getAs[Long]("stateIndex") == numSamples - 1)(0)
          .getAs[DenseVector]("state")

        val stats = modelState.groupBy($"stateKey")
          .agg(
            Summarizer.mean($"state").alias("avg"))
          .head
        assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - zs.reduce(_ + _)/zs.size) < 1.0)
      }

      it("should have same result for batch & stream mode") {
        testAppendQueryAgainstBatch(measurements, query, "dynamicLinearModelSmoothing")
      }
    }
  }
}
