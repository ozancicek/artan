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

import com.github.ozancicek.artan.ml.mixture.MultivariateGaussianMixture
import com.github.ozancicek.artan.ml.SparkFunctions.{randMultiGaussian}
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


/**
 * Gaussian mixture model with streaming rate source
 *
 * To run the sample from source, build the assembly jar for artan-examples project and run:
 *
 * `spark-submit --class com.github.ozancicek.artan.examples.streaming.GMMRateSource artan-examples-assembly-VERSION.jar 10 10`
 */
object GMMRateSource {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: GMMRateSource <numStates> <measurementsPerSecond>")
      System.exit(1)
    }

    val spark = SparkSession
      .builder
      .appName("GMMRateSource")
      .getOrCreate
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val numStates = args(0).toInt
    val rowsPerSecond = args(1).toInt
    val numMixtures = 3
    val minibatchSize = 1
    // Define sample generating expression, 3 gaussians and a uniform random for mixture

    val dist1 = randMultiGaussian(new DenseVector(Array(1.0, 2.0)), DenseMatrix.eye(2), seed=0)
    val dist2 = randMultiGaussian(new DenseVector(Array(10.0, 5.0)), new DenseMatrix(2, 2, Array(4, 2, 2, 4)), seed=1)
    val dist3 = randMultiGaussian(new DenseVector(Array(4.0, 4.0)), new DenseMatrix(2, 2, Array(5, 0, 0, 5)), seed=2)

    // For mixture weights defined as [0.2, 0,3, 0.5], sample from uniform dist
    val weight = rand(seed=0)
    val mixture = when(weight < 0.2, dist1)
      .when(weight < 0.5, dist2)
      .otherwise(dist3)

    val eye = Array(1.0, 0.0, 0.0, 1.0)
    val gmm = new MultivariateGaussianMixture(3)
      .setStateKeyCol("stateKey")
      .setInitialMeans(Array(Array(3.0, 5.0), Array(6.0, 6.0), Array(7.0, 1.0)))
      .setInitialCovariances(Array(eye, eye, eye))
      .setStepSize(0.01)
      .setMinibatchSize(minibatchSize)

    val inputDf = spark.readStream.format("rate").option("rowsPerSecond", rowsPerSecond).load
      .withColumn("mod", $"value" % numStates)
      .withColumn("stateKey", $"mod".cast("String"))
      .withColumn("sample", mixture)

    // Helper udf to pretty print dense vectors & arrays
    val floor = (in: Double) => (math floor in * 100)/100
    val truncateVector = udf((in: DenseVector) => in.values.map(floor))
    val truncateArray= udf((in: Seq[Double]) => in.map(floor))

    val query = gmm.transform(inputDf)
      .select(
        $"stateKey", $"stateIndex", $"mixtureModel.weights",
        $"mixtureModel.distributions".getItem(0).alias("dist1"),
        $"mixtureModel.distributions".getItem(1).alias("dist2"),
        $"mixtureModel.distributions".getItem(2).alias("dist3"))
      .withColumn("weights", truncateArray($"weights"))
      .withColumn("dist1_mean", truncateVector($"dist1.mean"))
      .withColumn("dist2_mean", truncateVector($"dist2.mean"))
      .withColumn("dist3_mean", truncateVector($"dist3.mean"))
      .drop("dist1", "dist2", "dist3")
      .writeStream
      .queryName("GMMRateSource")
      .outputMode("append")
      .format("console")
      .start()

    query.awaitTermination()

    /*
    Batch: 1
    -------------------------------------------
    +--------+----------+------------------+------------+------------+------------+
    |stateKey|stateIndex|           weights|  dist1_mean|  dist2_mean|  dist3_mean|
    +--------+----------+------------------+------------+------------+------------+
    |       0|         1|[0.34, 0.33, 0.32]| [2.1, 5.36]|[6.38, 5.86]|[7.11, 1.03]|
    |       0|         2|[0.33, 0.33, 0.31]| [1.89, 5.2]|[6.24, 6.04]| [6.9, 0.98]|
    |       0|         3| [0.37, 0.31, 0.3]|[3.59, 4.98]| [6.4, 6.09]|[6.85, 1.02]|
    |       1|         1| [0.3, 0.37, 0.32]| [3.0, 4.99]| [7.46, 6.6]|[6.78, 0.84]|
    |       1|         2| [0.27, 0.42, 0.3]| [3.0, 4.99]|[8.14, 6.81]|[6.62, 0.86]|
    |       1|         3|[0.24, 0.42, 0.32]| [3.0, 4.99]|[7.61, 6.71]|[6.85, 0.95]|
    +--------+----------+------------------+------------+------------+------------+

    -------------------------------------------
    Batch: 2
    -------------------------------------------
    +--------+----------+------------------+------------+------------+------------+
    |stateKey|stateIndex|           weights|  dist1_mean|  dist2_mean|  dist3_mean|
    +--------+----------+------------------+------------+------------+------------+
    |       0|         4|[0.43, 0.28, 0.27]|[5.42, 5.96]|[6.44, 6.11]|[6.85, 1.02]|
    |       0|         5|[0.45, 0.28, 0.25]|[6.36, 6.44]|[6.61, 6.24]|[6.71, 0.99]|
    |       1|         4| [0.21, 0.47, 0.3]|[2.99, 4.98]| [6.95, 6.0]|[6.98, 1.03]|
    |       1|         5| [0.19, 0.49, 0.3]|[2.99, 4.98]|[6.98, 6.41]|[7.11, 1.16]|
    +--------+----------+------------------+------------+------------+------------+

    -------------------------------------------
    +--------+----------+------------------+------------+------------+-------------+
    |stateKey|stateIndex|           weights|  dist1_mean|  dist2_mean|   dist3_mean|
    +--------+----------+------------------+------------+------------+-------------+
    |       0|        97|[0.23, 0.38, 0.37]|[1.02, 2.03]|[3.93, 3.05]|  [9.94, 5.1]|
    |       1|        97| [0.15, 0.4, 0.43]|[0.93, 1.97]| [2.7, 3.79]|[10.52, 5.08]|
    +--------+----------+------------------+------------+------------+-------------+*/
  }
}