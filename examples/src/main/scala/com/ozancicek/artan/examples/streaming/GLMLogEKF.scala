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

package com.ozancicek.artan.examples.streaming

import com.ozancicek.artan.ml.filter.ExtendedKalmanFilter

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._

/**
 * Extended Kalman Filter example for solving GLM, gaussian noise with log link.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.ozancicek.artan.examples.batch.GLMLogEKF artan-examples-assembly-VERSION.jar 10 10`
 */
object GLMLogEKF {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: RateSourceLKF <numStates> <measurementsPerSecond>")
      System.exit(1)
    }

    val spark = SparkSession
      .builder
      .appName("GLMLogEKF")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt

    val noiseParam = 1.0

    // GLM with log link, states to be estimated are a, b
    // y = exp(a*x + b) + w, where w ~ N(0, 1)
    val a = 0.2
    val b = 0.7

    val measurementFunc = (in: Vector, model: Matrix) => {
      val measurement = model.multiply(in)
      measurement.values(0) = scala.math.exp(measurement.values(0))
      measurement
    }

    val measurementJac = (in: Vector, model: Matrix) => {
      val dot = model.multiply(in)
      val res = scala.math.exp(dot(0))
      val jacs = Array(
        model(0, 0) * res,
        res
      )
      new DenseMatrix(1, 2, jacs.toArray)
    }

    val filter = new ExtendedKalmanFilter(2, 1)
      .setStateKeyCol("stateKey")
      .setInitialCovariance(
        new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
      .setMeasurementCol("measurement")
      .setMeasurementModelCol("measurementModel")
      .setProcessModel(DenseMatrix.eye(2))
      .setProcessNoise(new DenseMatrix(2, 2, Array(0.00, 0.0, 0.0, 0.00)))
      .setMeasurementNoise(new DenseMatrix(1, 1, Array(10)))
      .setMeasurementFunction(measurementFunc)
      .setMeasurementStateJacobian(measurementJac)
      .setCalculateMahalanobis

    // UDF's for generating measurement vector ([y]) and measurement model matrix ([[x ,1]])
    val measurementUDF = udf((x: Double, r: Double) => {
      val measurement = scala.math.exp(a * x + b) + r
      new DenseVector(Array(measurement))
    })

    val measurementModelUDF = udf((x: Double) => {
      new DenseMatrix(1, 2, Array(x.toDouble, 1.0))
    })

    val measurements = spark.readStream.format("rate")
      .option("rowsPerSecond", rowsPerSecond)
      .load()
      .withColumn("mod", $"value" % numStates)
      .withColumn("stateKey", $"mod".cast("String"))
      .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
      .withColumn("measurement", measurementUDF($"x", randn() * noiseParam))
      .withColumn("measurementModel", measurementModelUDF($"x"))

    val query = filter.transform(measurements)
      .writeStream
      .queryName("GLMLogEKF")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()
  }
}