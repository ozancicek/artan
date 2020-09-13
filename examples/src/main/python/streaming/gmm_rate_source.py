#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import sys

from artan.mixture import MultivariateGaussianMixture
from artan.spark_functions import randnMultiGaussian

from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np


if __name__ == "__main__":
    """
    Gaussian mixture model with streaming rate source
    
    To run the sample from source, build the assembly jar for artan scala project, zip the artan python package
    and run:
    
    `spark-submit --py-files artan.zip --jars artan-examples-assembly-VERSION.jar gmm_rate_source.py 2 10`
    """
    if len(sys.argv) != 3:
        print("Usage: gmm_rate_source.py <num_states> <measurements_per_sec>", file=sys.stderr)
        sys.exit(-1)

    num_states = int(sys.argv[1])
    mps = int(sys.argv[2])

    spark = SparkSession.builder.appName("GMMRateSource").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Define sample generating expression, 3 gaussians and a uniform random for mixture weights
    num_mixtures = 3
    dist1 = randnMultiGaussian(np.array([1.0, 2.0]), np.eye(2), seed=0)
    dist2 = randnMultiGaussian(np.array([10.0, 5.0]), np.eye(2)*2 + 2, seed=1)
    dist3 = randnMultiGaussian(np.array([4.0, 4.0]), np.eye(2)*5, seed=2)

    weight = F.rand(seed=0)
    mixture = F\
        .when(weight < 0.2, dist1)\
        .when(weight < 0.5, dist2)\
        .otherwise(dist3)

    # Generate measurements for multiple models by modding the incremental $"value" column
    input_df = spark.readStream.format("rate").option("rowsPerSecond", mps).load()\
        .withColumn("mod", F.col("value") % num_states)\
        .withColumn("stateKey", F.col("mod").cast("String"))\
        .withColumn("sample", mixture)

    # Set parameters of the mixture model, with different means and identity cov matrices.
    minibatch_size = 1
    initial_means = [[3.0, 5.0], [6.0, 6.0], [7.0, 1.0]]
    eye = [1.0, 0.0, 0.0, 1.0]
    initial_covs = [eye, eye, eye]
    gmm = MultivariateGaussianMixture()\
        .setMixtureCount(3)\
        .setInitialWeights([0.33, 0.33, 0.33])\
        .setStateKeyCol("stateKey")\
        .setInitialMeans(initial_means)\
        .setInitialCovariances(initial_covs)\
        .setStepSize(0.01)\
        .setMinibatchSize(minibatch_size)

    # Helper udfs to pretty print vectors to console
    truncate_weights = F.udf(lambda x: "[%.2f, %.2f, %.2f]" % (x[0], x[1], x[2]), StringType())
    truncate_mean = F.udf(lambda x: "[%.2f, %.2f]" % (x[0], x[1]), StringType())

    # Run the transformer, extract estimated means from mixtureModel struct.
    query = gmm.transform(input_df)\
        .select(
            "stateKey", "stateIndex", "mixtureModel.weights",
            F.col("mixtureModel.distributions").getItem(0).alias("dist1"),
            F.col("mixtureModel.distributions").getItem(1).alias("dist2"),
            F.col("mixtureModel.distributions").getItem(2).alias("dist3"))\
        .withColumn("weights", truncate_weights("weights"))\
        .withColumn("dist1_mean", truncate_mean("dist1.mean"))\
        .withColumn("dist2_mean", truncate_mean("dist2.mean"))\
        .withColumn("dist3_mean", truncate_mean("dist3.mean"))\
        .drop("dist1", "dist2", "dist3")\
        .writeStream\
        .queryName("GMMRateSource")\
        .outputMode("append")\
        .format("console")\
        .start()

    """
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
    +--------+----------+------------------+------------+------------+------------+
        
    """
    query.awaitTermination()