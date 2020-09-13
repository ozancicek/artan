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
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.EKFRateSourceGLMLog artan-examples-assembly-VERSION.jar 2 2`
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

    val filter = new ExtendedKalmanFilter()
      .setStateKeyCol("modelID")
      .setInitialStateMean(new DenseVector(Array(0.0, 0.0)))
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
      .withColumn("modelID", $"mod".cast("String"))
      .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
      .withColumn("measurement", measurementUDF($"x", randn() * noiseParam))
      .withColumn("measurementModel", measurementModelUDF($"x"))

    val query = filter.transform(measurements)
      .select(
        $"modelID", $"stateIndex",
        $"state.mean".alias("stateMean"), $"state.covariance".alias("stateCovariance"),
        $"residual.mean".alias("residualMean"), $"residual.covariance".alias("residualCovariance"),
        $"mahalanobis")
      .writeStream
      .queryName("EKFRateSourceGLMLog")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()

  }
  /**
   * -------------------------------------------
   * Batch: 2
   * -------------------------------------------
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
   * |modelID|stateIndex|           stateMean|     stateCovariance|        residualMean|  residualCovariance|         mahalanobis|
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
   * |      0|         5|[-0.0170639651961...|0.184650735418856...|[-0.0010775678634...| 21.24279669719657  |2.337969194146342E-4|
   * |      0|         6|[0.13372113418410...|0.097270109221418...|[2.3866966781327466]|21.892368858374287  |  0.5100947459174262|
   * |      1|         5|[0.21727975764867...|0.184289044729487...|[2.1590034862902434]| 20.72475537603141  | 0.47425141689857636|
   * |      1|         6|[0.16619831285685...|0.061682057710189...|[-1.0041419082389...|47.378255003177436  | 0.14588329445602757|
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
   *
   * -------------------------------------------
   * Batch: 3
   * -------------------------------------------
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
   * |modelID|stateIndex|           stateMean|     stateCovariance|        residualMean|  residualCovariance|         mahalanobis|
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
   * |      0|         7|[0.21489917361592...|0.033224082430061...|[2.0552241094850023]| 41.05191755271204  | 0.32076905295206193|
   * |      0|         8|[0.20921262270095...|0.013189448768817...|[-0.2695123923053...| 45.00295378232299  |0.040175216810467415|
   * |      1|         7|[0.18172674610899...|0.031522374731488...|[0.4671830982405272]| 27.29893710175946  | 0.08941579732539723|
   * |      1|         8|[0.19249146732117...|0.016052060247902...|[0.4615553206598477]|28.440753092452763  | 0.08654723860064477|
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
   *
   * -------------------------------------------
   * Batch: 4
   * -------------------------------------------
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
   * |modelID|stateIndex|           stateMean|     stateCovariance|        residualMean|  residualCovariance|        mahalanobis|
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
   * |      0|         9|[0.18171784672603...|0.007654793457034...|[-1.9635172993212...| 28.22667169246637  |0.36957696607714374|
   * |      1|         9|[0.17499288278196...|0.008676615020153...|[-1.070230083612481]|27.589047780543666  |0.20375524590073577|
   * +-------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
   */
}