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

import com.github.ozancicek.artan.ml.filter.ExtendedKalmanFilter
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._

/**
 * Extended Kalman Filter example for solving GLM, gaussian noise with log link.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.EKFRateSourceGLMLog artan-examples-assembly-VERSION.jar 10 10`
 */
object EKFRateSourceGLMLog {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: EKFRateSourceGLMLog <numStates> <measurementsPerSecond>")
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
      new DenseMatrix(1, 2, jacs)
    }

    val filter = new ExtendedKalmanFilter(stateSize, measurementSize)
      .setStateKeyCol("stateKey")
      .setInitialStateCovariance(
        new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
      .setMeasurementCol("measurement")
      .setMeasurementModelCol("measurementModel")
      .setProcessModel(DenseMatrix.eye(2))
      .setProcessNoise(DenseMatrix.zeros(2, 2))
      .setMeasurementNoise(new DenseMatrix(1, 1, Array(10)))
      .setMeasurementFunction(measurementFunc)
      .setMeasurementStateJacobian(measurementJac)
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
      .queryName("EKFRateSourceGLMLog")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()

  }
  /**
   * -------------------------------------------
   * Batch: 1
   * -------------------------------------------
   * +--------+----------+--------------------+--------------------+--------------------+
   * |stateKey|stateIndex|               state|            residual|         mahalanobis|
   * +--------+----------+--------------------+--------------------+--------------------+
   * |       0|         1|[[0.0,-0.36700349...|[[-0.734006980821...| 0.16412895050767087|
   * |       0|         2|[[0.6645974681322...|[[1.6499388441441...| 0.39783873028737665|
   * |       0|         3|[[0.4931085153689...|[[-1.360296026962...| 0.07006118446723411|
   * |       0|         4|[[0.3537506237758...|[[-1.192503763914...| 0.15990487547313653|
   * |       1|         1|[[0.0,0.938667451...|[[1.8773349027558...| 0.41978484590950504|
   * |       1|         2|[[-0.178414896203...|[[-0.753982653536...| 0.07253826868305598|
   * |       1|         3|[[-0.148700557485...|[[0.1089449625335...|0.020141106241910633|
   * |       1|         4|[[-0.012099196739...|[[0.7120366501275...| 0.13850897492065434|
   * +--------+----------+--------------------+--------------------+--------------------+
   *
   * -------------------------------------------
   * Batch: 2
   * -------------------------------------------
   * +--------+----------+--------------------+--------------------+-------------------+
   * |stateKey|stateIndex|               state|            residual|        mahalanobis|
   * +--------+----------+--------------------+--------------------+-------------------+
   * |       0|         5|[[0.4798014296080...|[[1.5566727142987...|0.23808929520642347|
   * |       0|         6|[[0.2776671140084...|[[-5.496350143076...| 0.5693359036120668|
   * |       0|         7|[[0.1826767679238...|[[-3.177261323635...| 0.6564003568723662|
   * |       1|         5|[[0.0939583624501...|[[0.8778806203227...|  0.150063577873534|
   * |       1|         6|[[0.2294841000282...|[[1.9725404812319...| 0.3138351703432594|
   * |       1|         7|[[0.2447102851239...|[[0.4559329677756...|0.05712734979579237|
   * +--------+----------+--------------------+--------------------+-------------------+
   */
}