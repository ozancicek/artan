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
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._


/**
 * Kalman filter local linear trend, initialize with batch dataframe and continue with stream dataframe
 * local performance & capacity testing.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.LKFStreamBatchInit artan-examples-assembly-VERSION.jar 10 10`
 */
object LKFStreamBatchInit {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: LKFStreamBatchInit <numStates> <measurementsPerSecond>")
      System.exit(1)
    }
    val spark = SparkSession
      .builder
      .appName("LKFStreamBatchInit")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt

    val noiseParam = 1.0

    val measurementUdf = udf((t: Long, r: Double) => new DenseVector(Array(t.toDouble + r)))

    val batchFilter = new LinearKalmanFilter(2, 1)
      .setStateKeyCol("stateKey")
      .setMeasurementCol("measurement")
      .setInitialStateCovariance(
        new DenseMatrix(2, 2, Array(1000, 0, 0, 1000)))
      .setProcessModel(
        new DenseMatrix(2, 2, Array(1, 0, 1, 1)))
      .setProcessNoise(
        new DenseMatrix(2, 2, Array(0.0001, 0.0, 0.0, 0.0001)))
      .setMeasurementNoise(
        new DenseMatrix(1, 1, Array(noiseParam)))
      .setMeasurementModel(
        new DenseMatrix(1, 2, Array(1, 0)))

    // Function to generate measurements dataframe
    val generateMeasurements = (df: DataFrame) => {
      df.withColumn("mod", $"value" % numStates)
        .withColumn("stateKey", $"mod".cast("String"))
        .withColumn("measurement", measurementUdf($"value"/numStates, randn() * noiseParam))
    }

    val batchMeasurementCount = 10 * rowsPerSecond
    val batchMeasurements = generateMeasurements((0 to numStates * batchMeasurementCount).toDF("value"))
    // Get the latest state from the filter trained in batch mode.
    val batchState = batchFilter.transform(batchMeasurements)
      .filter(s"stateIndex = $batchMeasurementCount")
      .select("stateKey", "state").cache()
    batchState.show(numStates)

    // Copy batch filter, except initial state is read from dataframe column
    val streamFilter = batchFilter
      .setInitialStateDistributionCol("state")

    // Generate streaming DF, shift the value by batchMeasurementCount to remove overlap with batch train data
    val streamDF = generateMeasurements(spark.readStream.format("rate")
      .option("rowsPerSecond", rowsPerSecond)
      .load()
      .withColumn("value", $"value" + numStates * batchMeasurementCount))

    // Static-stream join to add state & stateCovariance columns.
    val streamMeasurements = streamDF
      .join(batchState, "stateKey")

    val query = streamFilter.transform(streamMeasurements)
      .writeStream
      .queryName("LKFStreamBatchInit")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()
  }

  /**
   * +--------+--------------------+
   * |stateKey|               state|
   * +--------+--------------------+
   * |       0|[[19.515662167212...|
   * |       1|[[19.468300376746...|
   * +--------+--------------------+
   *
   * -------------------------------------------
   * Batch: 0
   * -------------------------------------------
   * +--------+----------+-----+
   * |stateKey|stateIndex|state|
   * +--------+----------+-----+
   * +--------+----------+-----+
   *
   * -------------------------------------------
   * Batch: 1
   * -------------------------------------------
   * +--------+----------+--------------------+
   * |stateKey|stateIndex|               state|
   * +--------+----------+--------------------+
   * |       0|         1|[[20.633594973582...|
   * |       0|         2|[[21.789119165418...|
   * |       0|         3|[[22.863165204359...|
   * |       1|         1|[[20.455445849573...|
   * |       1|         2|[[21.844582344505...|
   * |       1|         3|[[22.451603629903...|
   * +--------+----------+--------------------+
   *
   * -------------------------------------------
   * Batch: 2
   * -------------------------------------------
   * +--------+----------+--------------------+
   * |stateKey|stateIndex|               state|
   * +--------+----------+--------------------+
   * |       0|         4|[[23.906030752251...|
   * |       0|         5|[[25.025250483006...|
   * |       1|         4|[[23.195987426484...|
   * |       1|         5|[[24.199653312063...|
   * +--------+----------+--------------------+
   */
}