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

from artan.filter import LinearKalmanFilter

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, Matrices


if __name__ == "__main__":
    """
    Continuously filters a local linear increasing trend with a rate source, primarily for a quick
    local performance & capacity testing.
    To run the sample from source, build the assembly jar for artan scala project, zip the artan python package
    and run:
    
    `spark-submit --py-files artan.zip --jars artan-examples-assembly-VERSION.jar rate_source_lkf.py 1000 1000`
    """
    if len(sys.argv) != 3:
        print("Usage: rate_source_lkf.py <num_states> <measurements_per_sec>", file=sys.stderr)
        sys.exit(-1)

    num_states = int(sys.argv[1])
    mps = int(sys.argv[2])

    spark = SparkSession.builder.appName("RateSourceLKF").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    noise_param = 20

    input_df = spark.readStream.format("rate").option("rowsPerSecond", mps).load()\
        .withColumn("stateKey", (F.col("value") % num_states).cast("String"))\
        .withColumn("trend", F.col("value") + F.randn() * noise_param)

    lkf = LinearKalmanFilter(2, 1)\
        .setStateKeyCol("stateKey")\
        .setMeasurementCol("measurement")\
        .setInitialCovariance(Matrices.dense(2, 2, [10000.0, 0.0, 0.0, 10000.0]))\
        .setProcessModel(Matrices.dense(2, 2, [1.0, 0.0, 1.0, 1.0]))\
        .setProcessNoise(Matrices.dense(2, 2, [0.0001, 0.0, 0.0, 0.0001]))\
        .setMeasurementNoise(Matrices.dense(1, 1, [noise_param]))\
        .setMeasurementModel(Matrices.dense(1, 2, [1.0, 0.0]))

    assembler = VectorAssembler(inputCols=["trend"], outputCol="measurement")

    measurements = assembler.transform(input_df)
    query = lkf.transform(measurements)\
        .writeStream\
        .queryName("RateSourceLKF")\
        .outputMode("append")\
        .format("console")\
        .start()

    query.awaitTermination()