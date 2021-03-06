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
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._


/**
 * Continuously filters a local linear increasing trend with a rate source, primarily for a quick
 * local performance & capacity testing.
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.LKFRateSourceLLT artan-examples-assembly-VERSION.jar 10 10`
 */
object LKFRateSourceLLT {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: LKFRateSourceLLT <numStates> <measurementsPerSecond>")
      System.exit(1)
    }
    val spark = SparkSession
      .builder
      .appName("LLTRateSourceLKF")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt

    val noiseParam = 1.0

    val measurementUdf = udf((t: Long, r: Double) => new DenseVector(Array(t.toDouble + r)))

    val filter = new LinearKalmanFilter()
      .setStateKeyCol("stateKey")
      .setMeasurementCol("measurement")
      .setInitialStateMean(new DenseVector(Array(0.0, 0.0)))
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

    val measurements = spark.readStream.format("rate")
      .option("rowsPerSecond", rowsPerSecond)
      .load()
      .withColumn("mod", $"value" % numStates)
      .withColumn("stateKey", $"mod".cast("String"))
      .withColumn("measurement", measurementUdf($"value"/numStates, randn() * noiseParam))

    val query = filter.transform(measurements)
      .writeStream
      .queryName("LKFRateSourceLLT")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()
  }

  /**
   * -------------------------------------------
   * Batch: 1
   * -------------------------------------------
   * +--------+----------+--------------------+
   * |stateKey|stateIndex|               state|
   * +--------+----------+--------------------+
   * |       0|         1|[[0.1492418934476...|
   * |       0|         2|[[-0.321771094817...|
   * |       0|         3|[[0.6882290382067...|
   * |       0|         4|[[3.7839664559790...|
   * |       1|         1|[[-0.398664215642...|
   * |       1|         2|[[1.8770573391696...|
   * |       1|         3|[[3.2435865142240...|
   * |       1|         4|[[3.3700889676973...|
   * +--------+----------+--------------------+
   *
   * -------------------------------------------
   * Batch: 2
   * -------------------------------------------
   * +--------+----------+--------------------+
   * |stateKey|stateIndex|               state|
   * +--------+----------+--------------------+
   * |       0|         5|[[4.0174424704329...|
   * |       0|         6|[[5.2113814990445...|
   * |       0|         7|[[5.2297892651594...|
   * |       1|         5|[[4.7256361837408...|
   * |       1|         6|[[4.6242844388232...|
   * |       1|         7|[[6.0334517144475...|
   * +--------+----------+--------------------+
   */
}