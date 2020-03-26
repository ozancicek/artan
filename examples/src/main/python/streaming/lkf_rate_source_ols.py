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
from pyspark.ml.linalg import Matrices, Vectors, MatrixUDT, VectorUDT
from pyspark.sql.types import StringType


if __name__ == "__main__":
    """
    Recursive Least Squares with streaming rate source
    
    To run the sample from source, build the assembly jar for artan scala project, zip the artan python package
    and run:
    
    `spark-submit --py-files artan.zip --jars artan-examples-assembly-VERSION.jar lkf_rate_source_ols.py 2 10`
    """
    if len(sys.argv) != 3:
        print("Usage: lkf_rate_source_ols.py <num_states> <measurements_per_sec>", file=sys.stderr)
        sys.exit(-1)

    num_states = int(sys.argv[1])
    mps = int(sys.argv[2])

    spark = SparkSession.builder.appName("LKFRateSourceOLS").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # OLS problem, states to be estimated are a, b and c
    # z = a*x + b * y + c + w, where w ~ N(0, 1)
    a = 0.5
    b = 0.2
    c = 1.2
    noise_param = 1
    state_size = 3
    measurement_size = 1

    label_udf = F.udf(lambda x, y, w: Vectors.dense([x * a + y * b + c + w]), VectorUDT())
    features_udf = F.udf(lambda x, y: Matrices.dense(1, 3, [x, y, 1]), MatrixUDT())

    features = spark.readStream.format("rate").option("rowsPerSecond", mps).load()\
        .withColumn("mod", F.col("value") % num_states)\
        .withColumn("stateKey", F.col("mod").cast("String"))\
        .withColumn("x", (F.col("value")/num_states).cast("Integer").cast("Double"))\
        .withColumn("y", F.sqrt("x"))\
        .withColumn("w", F.randn(0) * noise_param)\
        .withColumn("label", label_udf("x", "y", "w"))\
        .withColumn("features", features_udf("x", "y"))

    lkf = LinearKalmanFilter(state_size, measurement_size)\
        .setStateKeyCol("stateKey")\
        .setMeasurementCol("label")\
        .setMeasurementModelCol("features")\
        .setInitialCovariance(Matrices.dense(3, 3, [10, 0, 0, 0, 10, 0, 0, 0, 10]))\
        .setProcessModel(Matrices.dense(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]))\
        .setProcessNoise(Matrices.dense(3, 3, [0] * 9))\
        .setMeasurementNoise(Matrices.dense(1, 1, [1]))

    truncate_udf = F.udf(lambda x: "[%.2f, %.2f, %.2f]" % (x[0], x[1], x[2]), StringType())

    query = lkf.transform(features)\
        .select("stateKey", "stateIndex", truncate_udf("state").alias("modelParameters"))\
        .writeStream\
        .queryName("LKFRateSourceOLS")\
        .outputMode("append")\
        .format("console")\
        .start()

    query.awaitTermination()