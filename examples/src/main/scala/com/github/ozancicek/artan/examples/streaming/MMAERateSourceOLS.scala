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

import com.github.ozancicek.artan.ml.filter.LinearKalmanFilter
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


/**
 * Recursive Least Squares, multiple-model adaptive estimation with a bank of kalman filters & streaming rate source.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.MMAERateSourceOLS artan-examples-assembly-VERSION.jar 10 10`
 */
object MMAERateSourceOLS {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: MMAERateSourceOLS <numStates> <measurementsPerSecond>")
      System.exit(1)
    }
    val spark = SparkSession
      .builder
      .appName("MMAERateSourceOLS")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt

    // OLS problem, states to be estimated are a, b and c
    // z = a*x + b * y + c + w, where w ~ N(0, 1)
    val a = 0.5
    val b = 0.2
    val c = 1.2
    val stateSize = 3
    val measurementsSize = 1
    val noiseParam = 1.0

    val featuresUDF = udf((x: Double, y: Double) => {
      new DenseMatrix(measurementsSize, stateSize, Array(x, y, 1.0))
    })

    val labelUDF = udf((x: Double, y: Double, r: Double) => {
      new DenseVector(Array(a*x + b*y + c + r))
    })

    val features = spark.readStream.format("rate")
      .option("rowsPerSecond", rowsPerSecond)
      .load()
      .withColumn("mod", $"value" % numStates)
      .withColumn("stateKey", $"mod".cast("String"))
      .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
      .withColumn("y", sqrt($"x"))
      .withColumn("label", labelUDF($"x", $"y", randn() * noiseParam))
      .withColumn("features", featuresUDF($"x", $"y"))

    val filter = new LinearKalmanFilter()
      .setInitialStateMean(new DenseVector(Array(0.0, 0.0, 0.0)))
      .setInitialStateCovariance(
        new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
      .setStateKeyCol("stateKey")
      .setMeasurementCol("label")
      .setMeasurementModelCol("features")
      .setProcessModel(DenseMatrix.eye(stateSize))
      .setProcessNoise(DenseMatrix.zeros(stateSize, stateSize))
      .setMeasurementNoise(DenseMatrix.eye(measurementsSize))
      .setSlidingLikelihoodWindow(10)
      .setEventTimeCol("timestamp")
      .setWatermarkDuration("2 seconds")
      .setEnableMultipleModelAdaptiveEstimation
      .setMultipleModelMeasurementWindowDuration("5 seconds")

    val truncate = udf((state: DenseVector) => state.values.map(t => (math floor t * 100)/100))

    val query = filter.transform(features)
      .select( $"stateIndex", truncate($"state.mean").alias("modelParameters"), $"timestamp")
      .writeStream
      .queryName("MMAERateSourceOLS")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()
  }

  /**
   * -------------------------------------------
   * Batch: 3
   * -------------------------------------------
   * +----------+-------------------+--------------------+
   * |stateIndex|    modelParameters|           timestamp|
   * +----------+-------------------+--------------------+
   * |         5| [0.12, 0.51, 1.38]|[2020-06-29 18:37...|
   * |         3|[-0.35, 0.52, 1.75]|[2020-06-29 18:37...|
   * |         7|[0.71, -0.86, 2.16]|[2020-06-29 18:37...|
   * |         6| [0.34, 0.21, 1.37]|[2020-06-29 18:37...|
   * |         4| [0.49, 0.09, 1.34]|[2020-06-29 18:37...|
   * |         2| [0.39, 0.39, 1.26]|[2020-06-29 18:37...|
   * +----------+-------------------+--------------------+
   *
   * -------------------------------------------
   * Batch: 4
   * -------------------------------------------
   * +----------+-------------------+--------------------+
   * |stateIndex|    modelParameters|           timestamp|
   * +----------+-------------------+--------------------+
   * |         8|[0.91, -0.77, 1.43]|[2020-06-29 18:38...|
   * |         7|  [0.7, -0.32, 1.3]|[2020-06-29 18:38...|
   * |        10|[0.53, -0.04, 1.44]|[2020-06-29 18:38...|
   * |         9| [0.7, -0.42, 1.51]|[2020-06-29 18:38...|
   * |        12|[0.67, -0.56, 1.99]|[2020-06-29 18:38...|
   * |        11|[0.53, -0.03, 1.41]|[2020-06-29 18:38...|
   * +----------+-------------------+--------------------+
   *
   * -------------------------------------------
   * Batch: 5
   * -------------------------------------------
   * +----------+-------------------+--------------------+
   * |stateIndex|    modelParameters|           timestamp|
   * +----------+-------------------+--------------------+
   * |        16|[0.56, -0.16, 1.56]|[2020-06-29 18:38...|
   * |        15|[0.58, -0.21, 1.56]|[2020-06-29 18:38...|
   * |        13|  [0.51, 0.05, 1.3]|[2020-06-29 18:38...|
   * |        17|[0.62, -0.43, 1.98]|[2020-06-29 18:38...|
   * |        12|[0.54, -0.01, 1.28]|[2020-06-29 18:38...|
   * |        14|  [0.5, 0.02, 1.43]|[2020-06-29 18:38...|
   * +----------+-------------------+--------------------+
   */
}