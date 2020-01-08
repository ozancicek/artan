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

import com.ozancicek.artan.ml.testutils.SparkSessionTestWrapper
import org.scalatest.{FunSpec, Matchers}
import org.apache.spark.ml.linalg._
import breeze.stats.distributions.{RandBasis}
import org.apache.spark.ml.{LAPACK}
import scala.math.{exp, abs, sqrt}


class UnscentedKalmanFilterSpec
  extends FunSpec
  with Matchers
  with SparkSessionTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)

  describe("Batch unscented kalman filter tests") {
    it("should estimate linear regression model parameters") {
      // linear regression
      // z = a*x + b*y + c + N(0, 1)
      val a = 0.5
      val b = -0.7
      val c = 2.0
      val dist = breeze.stats.distributions.Gaussian(0, 1)
      val n = 40
      val xs = (0 until n).map(i=>i.toDouble).toArray
      val ys = (0 until n).map(i=> sqrt(i.toDouble)).toArray
      val zs = xs.zip(ys).map {
        case(x,y)=> (x, y, a*x + b*y + c + dist.draw())
      }
      val df = zs.map {
        case (x, y, z) => ("1", new DenseVector(Array(z)), new DenseMatrix(1, 3, Array(x, y, 1)))
      }.toSeq.toDF("modelId", "measurement", "measurementModel")

      val measurementFunc = (in: Vector, model: Matrix) => {
        val measurement = model.multiply(in)
        //measurement.values(0) = scala.math.exp(measurement.values(0))
        measurement
      }

      val filter = new UnscentedKalmanFilter(3, 1)
        .setGroupKeyCol("modelId")
        .setStateCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))
        .setMeasurementFunction(measurementFunc)

      val modelState = filter.transform(df)
      val lastState = modelState.collect
        .filter(row=>row.getAs[Long]("index") == n)(0)
        .getAs[DenseVector]("mean")

      val features = new DenseMatrix(n, 3, xs ++ ys ++ Array.fill(n) {1.0})
      val target = new DenseVector(zs.map {case (x, y, z) => z}.toArray)
      val coeffs = LAPACK.dgels(features, target)
      val mae = (0 until coeffs.size).foldLeft(0.0) {
        case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
      } / coeffs.size
      // Error should be smaller than a certain threshold. The threshold is
      // tuned to some arbitrary small value depending on noise, cov and true coefficients.
      val threshold = 1E-4
      assert(mae < threshold)
    }
  }

  it("should estimate glm with logit link") {
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

    val df = zs.map {
      case (x, y, z) => ("1", new DenseVector(Array(z)), new DenseMatrix(1, 3, Array(x, y, 1)))
    }.toSeq.toDF("modelId", "measurement", "measurementModel")

    val measurementFunc = (in: Vector, model: Matrix) => {
      val measurement = model.multiply(in)
      measurement.values(0) = logit(measurement.values(0))
      measurement
    }

    val filter = new UnscentedKalmanFilter(3, 1)
      .setGroupKeyCol("modelId")
      .setStateCovariance(
        new DenseMatrix(3, 3, Array(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)))
      .setMeasurementCol("measurement")
      .setMeasurementModelCol("measurementModel")
      .setProcessModel(DenseMatrix.eye(3))
      .setProcessNoise(DenseMatrix.zeros(3, 3))
      .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.001)))
      .setMeasurementFunction(measurementFunc)

    val modelState = filter.transform(df)

    val lastState = modelState.collect
      .filter(row=>row.getAs[Long]("index") == n)(0)
      .getAs[DenseVector]("mean")

    val coeffs = new DenseVector(Array(a, b, c))
    val mae = (0 until coeffs.size).foldLeft(0.0) {
      case(s, i) => s + abs(lastState(i) - coeffs(i))
    } / coeffs.size
    val threshold = 0.1
    assert(mae < threshold)
  }
}
