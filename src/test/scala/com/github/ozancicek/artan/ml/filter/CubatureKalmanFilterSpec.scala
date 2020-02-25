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


case class CKFOLSMeasurement(measurement: DenseVector, measurementModel: DenseMatrix)


class CubatureKalmanFilterSpec
  extends FunSpec
  with Matchers
  with RegressionTestWrapper {


  describe("Cubature kalman filter tests") {
    describe("Ordinary least squares") {

      val measurementFunc = (in: Vector, model: Matrix) => {
        val measurement = model.multiply(in)
        measurement
      }

      val filter = new CubatureKalmanFilter(3, 1)
        .setInitialCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))
        .setMeasurementFunction(measurementFunc)

      it("should have same solution with lapack dgels routine") {
        testLeastSquaresSolutionEquivalent(filter, 10E-3)
      }
    }

    describe("GLM with gaussian noise and log link") {
      val measurementFunc = (in: Vector, model: Matrix) => {
        val measurement = model.multiply(in)
        measurement.values(0) = scala.math.exp(measurement.values(0))
        measurement
      }

      val filter = new CubatureKalmanFilter(3, 1)
        .setInitialCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(10)))
        .setMeasurementFunction(measurementFunc)

      it("should estimate model parameters") {
        testLogRegressionEquivalent(filter, 10E-3)
      }

      it("should have same result for batch & stream mode") {
        testLogRegressionBatchStreamEquivalent(filter, "CKFLogResult")
      }
    }
  }
}
