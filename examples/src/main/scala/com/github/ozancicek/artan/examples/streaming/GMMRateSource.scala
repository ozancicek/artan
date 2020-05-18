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
import com.github.ozancicek.artan.ml.SparkFunctions.randMultiGaussian
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

    // Define 3 gaussians for sample generating expression
    val dist1 = randMultiGaussian(new DenseVector(Array(1.0, 2.0)), DenseMatrix.eye(2), seed=0)
    val dist2 = randMultiGaussian(new DenseVector(Array(10.0, 5.0)), new DenseMatrix(2, 2, Array(4, 2, 2, 4)), seed=1)
    val dist3 = randMultiGaussian(new DenseVector(Array(4.0, 4.0)), new DenseMatrix(2, 2, Array(5, 0, 0, 5)), seed=2)

    // Mixture weights defined as [0.2, 0,3, 0.5], sample from uniform dist
    val weight = rand(seed=0)
    val mixture = when(weight < 0.2, dist1).when(weight < 0.5, dist2).otherwise(dist3)

    val gmm = new MultivariateGaussianMixture(3)
      .setStateKeyCol("stateKey")
      .setInitialMeans(Array(Array(3.0, 5.0), Array(6.0, 6.0), Array(7.0, 1.0)))
      .setInitialCovariances(Array(Array(1.0, 0.0, 0.0, 1.0), Array(1.0, 0.0, 0.0, 1.0), Array(1.0, 0.0, 0.0, 1.0)))
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
    -------------------------------------------
    Batch: 1
    -------------------------------------------
    +--------+----------+------------------+------------+------------+------------+
    |stateKey|stateIndex|           weights|  dist1_mean|  dist2_mean|  dist3_mean|
    +--------+----------+------------------+------------+------------+------------+
    |       0|         1|[0.33, 0.33, 0.33]|[2.98, 4.97]|[6.00, 6.00]|[7.02, 1.02]|
    |       0|         2|[0.33, 0.33, 0.33]|[2.96, 4.95]|[6.03, 6.00]|[7.03, 1.04]|
    |       1|         1|[0.33, 0.33, 0.33]|[2.98, 4.99]|[6.02, 5.99]|[7.00, 1.01]|
    |       1|         2|[0.33, 0.33, 0.33]|[2.98, 4.97]|[6.06, 6.00]|[7.03, 1.02]|
    +--------+----------+------------------+------------+------------+------------+

    -------------------------------------------
    Batch: 2
    -------------------------------------------
    +--------+----------+------------------+------------+------------+------------+
    |stateKey|stateIndex|           weights|  dist1_mean|  dist2_mean|  dist3_mean|
    +--------+----------+------------------+------------+------------+------------+
    |       0|         3|[0.34, 0.33, 0.33]|[2.95, 4.91]|[6.10, 6.04]|[7.03, 1.04]|
    |       0|         4|[0.33, 0.34, 0.33]|[2.95, 4.91]|[6.13, 6.03]|[7.04, 1.06]|
    |       1|         3|[0.33, 0.33, 0.33]|[2.96, 4.97]|[6.08, 6.00]|[7.02, 1.02]|
    |       1|         4|[0.33, 0.33, 0.33]|[2.95, 4.95]|[6.13, 6.01]|[7.06, 1.04]|
    +--------+----------+------------------+------------+------------+------------+

    -------------------------------------------
    Batch: 10
    -------------------------------------------
    +--------+----------+------------------+------------+------------+------------+
    |stateKey|stateIndex|           weights|  dist1_mean|  dist2_mean|  dist3_mean|
    +--------+----------+------------------+------------+------------+------------+
    |       0|        16|[0.42, 0.45, 0.13]|[2.17, 3.59]|[9.05, 5.64]|[7.57, 1.49]|
    |       1|        16|[0.41, 0.30, 0.29]|[2.13, 3.35]|[7.79, 5.61]|[7.71, 1.96]|
    +--------+----------+------------------+------------+------------+------------+*/
  }
}