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

package com.github.ozancicek.artan.examples.streaming

import com.github.ozancicek.artan.ml.filter.UnscentedKalmanFilter
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Unscented Kalman Filter example for solving GLM, gaussian noise with log link.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.UKFRateSourceGLMLog artan-examples-assembly-VERSION.jar 10 10`
 */
object UKFRateSourceGLMLog {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: UKFRateSourceGLMLog <numStates> <measurementsPerSecond>")
      System.exit(1)
    }

    val spark = SparkSession
      .builder
      .appName("GLMLogUKF")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt

    // GLM with log link, states to be estimated are a, b
    // y = exp(a*x + b) + w, where w ~ N(0, 1)
    val a = 0.2
    val b = 0.7
    val noiseParam = 1.0
    val stateSize = 2
    val measurementSize = 1

    // UDF's for generating measurement vector ([y]) and measurement model matrix ([[x ,1]])
    val measurementUDF = udf((x: Double, r: Double) => {
      val measurement = scala.math.exp(a * x + b) + r
      new DenseVector(Array(measurement))
    })

    val measurementModelUDF = udf((x: Double) => {
      new DenseMatrix(1, 2, Array(x, 1.0))
    })

    // No jac func is needed compared to EKF
    val measurementFunc = (in: Vector, model: Matrix) => {
      val measurement = model.multiply(in)
      measurement.values(0) = scala.math.exp(measurement.values(0))
      measurement
    }

    val filter = new UnscentedKalmanFilter(stateSize, measurementSize)
      .setStateKeyCol("stateKey")
      .setInitialCovariance(
        DenseMatrix.eye(2))
      .setMeasurementCol("measurement")
      .setMeasurementModelCol("measurementModel")
      .setProcessModel(DenseMatrix.eye(2))
      .setProcessNoise(DenseMatrix.zeros(2, 2))
      .setMeasurementNoise(DenseMatrix.eye(1))
      .setMeasurementFunction(measurementFunc)
      .setCalculateMahalanobis

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
      .queryName("UKFRateSourceGLMLog")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()
  }
}