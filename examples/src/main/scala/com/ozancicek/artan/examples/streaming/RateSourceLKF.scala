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

import com.ozancicek.artan.ml.filter.LinearKalmanFilter

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._


/**
 * Continiously filters a local linear increasing trend with a rate source, primarily for a quick
 * local performance & capacity testing.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.ozancicek.artan.examples.streaming.RateSourceLKF artan-examples-assembly-VERSION.jar 10 10`
 */
object RateSourceLKF {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: RateSourceLKF <numStates> <measurementsPerSecond>")
      System.exit(1)
    }
    val spark = SparkSession
      .builder
      .appName("RateSourceLKF")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt

    val noiseParam = 20.0

    val measurementUdf = udf((t: Long, r: Double) => new DenseVector(Array(t.toDouble + r)))

    val filter = new LinearKalmanFilter(2, 1)
      .setStateKeyCol("stateKey")
      .setMeasurementCol("measurement")
      .setInitialCovariance(
        new DenseMatrix(2, 2, Array(1000, 0, 0, 1000)))
      .setProcessModel(
        new DenseMatrix(2, 2, Array(1, 0, 1, 1)))
      .setProcessNoise(
        new DenseMatrix(2, 2, Array(0.0001, 0.0, 0.0, 0.0001)))
      .setMeasurementNoise(
        new DenseMatrix(1, 1, Array(noiseParam)))
      .setMeasurementModel(
        new DenseMatrix(1, 2, Array(1, 0)))

    val measurements = spark.readStream.format("rate")
      .option("rowsPerSecond", rowsPerSecond)
      .load()
      .withColumn("measurement", measurementUdf($"value", randn() * noiseParam))
      .withColumn("stateKey", ($"value" % numStates).cast("String"))

    val query = filter.transform(measurements)
      .writeStream
      .queryName("RateSourceLKF")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()
  }
}