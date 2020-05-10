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

    num_mixtures = 3
    minibatch_size = 1
    dist1 = randnMultiGaussian(np.array([1.0, 2.0]), np.eye(2), seed=0)
    dist2 = randnMultiGaussian(np.array([10.0, 5.0]), np.eye(2)*2 + 2, seed=1)
    dist3 = randnMultiGaussian(np.array([4.0, 4.0]), np.eye(2)*5, seed=2)

    weight = F.rand(seed=0)
    mixture = F\
        .when(weight < 0.2, dist1)\
        .when(weight < 0.5, dist2)\
        .otherwise(dist3)

    input_df = spark.readStream.format("rate").option("rowsPerSecond", mps).load()\
        .withColumn("mod", F.col("value") % num_states)\
        .withColumn("stateKey", F.col("mod").cast("String"))\
        .withColumn("sample", mixture)

    eye = [1.0, 0.0, 0.0, 1.0]
    gmm = MultivariateGaussianMixture(3)\
        .setStateKeyCol("stateKey")\
        .setInitialMeans([[3.0, 5.0], [6.0, 6.0], [7.0, 1.0]])\
        .setInitialCovariances([eye, eye, eye])\
        .setStepSize(0.01)\
        .setMinibatchSize(minibatch_size)

    truncate_weights = F.udf(lambda x: "[%.2f, %.2f, %.2f]" % (x[0], x[1], x[2]), StringType())

    truncate_mean = F.udf(lambda x: "[%.2f, %.2f]" % (x[0], x[1]), StringType())

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

    query.awaitTermination()