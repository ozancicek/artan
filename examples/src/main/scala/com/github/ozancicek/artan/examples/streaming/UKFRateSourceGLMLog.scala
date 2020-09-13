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

    val filter = new UnscentedKalmanFilter()
      .setStateKeyCol("stateKey")
      .setInitialStateMean(new DenseVector(Array(0.0, 0.0)))
      .setInitialStateCovariance(
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
  /*

  -------------------------------------------
  Batch: 1
  -------------------------------------------
  +--------+----------+---------+--------------------+--------------------+-------------------+
  |stateKey|stateIndex|stepIndex|               state|            residual|        mahalanobis|
  +--------+----------+---------+--------------------+--------------------+-------------------+
  |       0|         1|        0|[[0.0,0.042083883...|[[0.1062957660590...|0.06584432237693608|
  |       0|         2|        0|[[-0.145367446951...|[[-6.532377989574...| 0.3788428065577319|
  |       0|         3|        0|[[0.1041022350732...|[[0.9724192360964...| 0.5627644378314648|
  |       0|         4|        0|[[0.2306342636805...|[[1.0403070080814...|  0.264478850278805|
  |       0|         5|        0|[[0.1063465161095...|[[-2.572317266578...| 0.3254493264520008|
  |       1|         1|        0|[[0.0,0.589622351...|[[1.4892722426408...| 0.9225214257075712|
  |       1|         2|        0|[[-0.204954508948...|[[-15.88844495110...| 0.5335348325127303|
  |       1|         3|        0|[[-0.118246670452...|[[0.4340293185286...|0.17419976429373318|
  |       1|         4|        0|[[0.1893660699514...|[[2.2655146428892...| 0.6555587631879435|
  |       1|         5|        0|[[-0.041656936742...|[[-5.686218438459...|  0.610362762749889|
  +--------+----------+---------+--------------------+--------------------+-------------------+

  -------------------------------------------
  Batch: 2
  -------------------------------------------
  +--------+----------+---------+--------------------+--------------------+--------------------+
  |stateKey|stateIndex|stepIndex|               state|            residual|         mahalanobis|
  +--------+----------+---------+--------------------+--------------------+--------------------+
  |       0|         6|        0|[[0.0535886042884...|[[-2.263659901210...|  0.1950673463505543|
  |       0|         7|        0|[[0.0234734490305...|[[-1.340034077244...|  0.1318700478960915|
  |       0|         8|        0|[[0.0672813961239...|[[1.9961239736383...| 0.22310849399461063|
  |       1|         6|        0|[[-0.041153233802...|[[0.0138346168400...|0.001907992146439...|
  |       1|         7|        0|[[0.0284426670012...|[[2.2444237109837...| 0.31476068999734547|
  |       1|         8|        0|[[0.0070700693878...|[[-1.172849191366...| 0.11280670334778167|
  +--------+----------+---------+--------------------+--------------------+--------------------+

  -------------------------------------------
  Batch: 3
  -------------------------------------------
  +--------+----------+---------+--------------------+--------------------+--------------------+
  |stateKey|stateIndex|stepIndex|               state|            residual|         mahalanobis|
  +--------+----------+---------+--------------------+--------------------+--------------------+
  |       0|         9|        0|[[0.0689014075795...|[[0.1174643793579...|0.009429133195598868|
  |       0|        10|        0|[[0.1079250363985...|[[3.2873115452680...|  0.2581465189858566|
  |       1|         9|        0|[[0.0851767804468...|[[4.0510944020424...|  0.4800965035588805|
  |       1|        10|        0|[[0.0976005513340...|[[1.2501195155840...| 0.08787380959737641|
  +--------+----------+---------+--------------------+--------------------+--------------------+


  -------------------------------------------
  Batch: 14
  -------------------------------------------
  +--------+----------+---------+--------------------+--------------------+-------------------+
  |stateKey|stateIndex|stepIndex|               state|            residual|        mahalanobis|
  +--------+----------+---------+--------------------+--------------------+-------------------+
  |       0|        25|        0|[[0.1979735110700...|[[-2.884962984114...| 1.6740538258342939|
  |       1|        25|        0|[[0.1996110381440...|[[-0.656398157846...|0.41074222096417723|
  +--------+----------+---------+--------------------+--------------------+-------------------+

  -------------------------------------------
  Batch: 15
  -------------------------------------------
  +--------+----------+---------+--------------------+--------------------+-------------------+
  |stateKey|stateIndex|stepIndex|               state|            residual|        mahalanobis|
  +--------+----------+---------+--------------------+--------------------+-------------------+
  |       0|        26|        0|[[0.1986075589232...|[[1.0770630811029...| 0.6503247720860308|
  |       1|        26|        0|[[0.1996956994673...|[[0.1826677631133...|0.11630923034534281|
  +--------+----------+---------+--------------------+--------------------+-------------------+

  */
}