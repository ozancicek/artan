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

package com.github.ozancicek.artan.ml.filter

import com.github.ozancicek.artan.ml.testutils.RegressionTestWrapper
import org.scalatest.{FunSpec, Matchers}
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.Dataset

import scala.math.{abs, exp, sqrt}


case class UKFMeasurement(measurement: DenseVector, measurementModel: DenseMatrix)


class UnscentedKalmanFilterSpec
  extends FunSpec
  with Matchers
  with RegressionTestWrapper {

  import spark.implicits._

  describe("Unscented kalman filter tests") {
    describe("Ordinary least squares") {

      val measurementFunc = (in: Vector, model: Matrix) => {
        val measurement = model.multiply(in)
        measurement
      }

      val filter = new UnscentedKalmanFilter(3, 1)
        .setInitialStateCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))
        .setMeasurementFunction(measurementFunc)


      it("should have same solution with lapack dgels routine") {
        testLeastSquaresSolutionEquivalent(filter, 10E-4)
      }

      it("should have same result for batch & stream mode") {
        testLeastSquaresBatchStreamEquivalent(filter, "UKFOls")
      }
    }
  }

  describe("GLM with logit link") {
    // glm with gaussian noise & logit link
    // z = logit(a*x + b*y + c) + N(0, R)

    val logit = (in: Double) => exp(in) / (1 + exp(in))
    val a = 0.2
    val b = -0.1
    val c = 0.2
    val dist = breeze.stats.distributions.Gaussian(0, 0.1)
    val n = 40
    val xs = (-n/2 until n/2).map(i=>i.toDouble).toArray
    val ys = (0 until n).map(i=> sqrt(i.toDouble)).toArray
    val zs = xs.zip(ys).map {
      case(x,y)=> (x, y, logit(a*x + b*y + c) + dist.draw())
    }

    val measurements = zs.map { case (x, y, z) =>
      UKFMeasurement(new DenseVector(Array(z)), new DenseMatrix(1, 3, Array(x, y, 1)))
    }.toSeq

    val measurementFunc = (in: Vector, model: Matrix) => {
      val measurement = model.multiply(in)
      measurement.values(0) = logit(measurement.values(0))
      measurement
    }

    val filter = new UnscentedKalmanFilter(3, 1)
      .setInitialStateCovariance(
        new DenseMatrix(3, 3, Array(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)))
      .setMeasurementCol("measurement")
      .setMeasurementModelCol("measurementModel")
      .setProcessModel(DenseMatrix.eye(3))
      .setProcessNoise(DenseMatrix.zeros(3, 3))
      .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.001)))
      .setMeasurementFunction(measurementFunc)
      .setSigmaPoints("merwe")
      .setMerweKappa(-0.7)
      .setCalculateMahalanobis

    val query = (in: Dataset[UKFMeasurement]) => filter.transform(in)

    it("should estimate model parameters") {
      val modelState = query(measurements.toDS)

      val lastState = modelState.select("state.mean", "stateIndex").collect
        .filter(row=>row.getAs[Long]("stateIndex") == n)(0)
        .getAs[DenseVector]("mean")

      val coeffs = new DenseVector(Array(a, b, c))
      val mae = (0 until coeffs.size).foldLeft(0.0) {
        case(s, i) => s + abs(lastState(i) - coeffs(i))
      } / coeffs.size
      val threshold = 0.1
      assert(mae < threshold)
    }

    it("should have same result for batch & stream mode") {
      testAppendQueryAgainstBatch(measurements, query, "UKFGLMModel")
    }
  }

  describe("Poisson process estimation") {

    val pm = new DenseMatrix(2, 2, Array(1.0, 0.0, 0.0, 1.0))
    val pn = new DenseMatrix(2, 2, Array(0.00001, 0.0, 0.0, 0.00001))
    val mn = new DenseMatrix(1, 1, Array(0.1))

    val modelFunc = (state: Vector, model: Matrix) => {
      val vals = model.toArray
      exp(vals(0)*state(0) + vals(1)*state(1))
    }

    val mFunc = (state: Vector, model: Matrix) => {
      new DenseVector(Array(modelFunc(state, model)))
    }

    val lbound = new DenseVector(Array(0.0, 0.0))
    val ubound = new DenseVector(Array(10.0, 10.0))

    val filter = new UnscentedKalmanFilter(2, 1)
      .setInitialStateMean(new DenseVector(Array(0.0, 0.0)))
      .setInitialStateCovariance(
        new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
      .setMeasurementCol("measurement")
      .setMeasurementModelCol("measurementModel")
      .setMeasurementNoise(mn)
      .setMeasurementFunction(mFunc)
      .setProcessModel(pm)
      .setProcessNoise(pn)
      .setSigmaPointLowerBound(lbound)
      .setSigmaPointUpperBound(ubound)

    val c1 = 2
    val c2 = 4

    val rates = Seq.fill(20) {1.0} ++ Seq.fill(50) {1.10} ++ Seq.fill(50) {0.95} ++ Seq.fill(150) {1.0}
    val measurements = rates.map { p=>
      val rate = exp(c1 - c2 * p)
      val obs = breeze.stats.distributions.Poisson(rate).draw()
      val measurement = new DenseVector(Array(obs))
      val mm = new DenseMatrix(1, 2, Array(1.0, -p))
      UKFMeasurement(measurement, mm)
    }.toDF()

    val modelState = filter.transform(measurements)
    it("should estimate parameters") {

      val lastState = modelState.select("state.mean", "stateIndex").collect
        .filter(row=>row.getAs[Long]("stateIndex") == rates.size)(0)
        .getAs[DenseVector]("mean")

      val coeffs = new DenseVector(Array(c1, c2))
      val mae = (0 until coeffs.size).foldLeft(0.0) {
        case(s, i) => s + abs(lastState(i) - coeffs(i))
      } / coeffs.size
      val threshold = 1.0
      assert(mae < threshold)
    }

    it("should stay within bounds") {
      modelState.select("state.mean").collect.foreach{ row=>
        val state = row.getAs[DenseVector]("mean")
        // Enforce lbound
        state.values.zip(lbound.values).foreach { case(s, lb) =>
          assert(s >= lb)
        }
        // Enforce ubound
        state.values.zip(ubound.values).foreach { case(s, ub) =>
          assert(s <= ub)
        }
      }
    }
  }
}
