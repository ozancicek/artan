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

from artan.filter import RecursiveLeastSquaresFilter

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler


if __name__ == "__main__":
    """
    Recursive Least Squares with streaming rate source
    
    To run the sample from source, build the assembly jar for artan scala project, zip the artan python package
    and run:
    
    `spark-submit --py-files artan.zip --jars artan-examples-assembly-VERSION.jar rls_rate_source_ols.py 2 10`
    """
    if len(sys.argv) != 3:
        print("Usage: rls_rate_source_ols.py <num_states> <measurements_per_sec>", file=sys.stderr)
        sys.exit(-1)

    num_states = int(sys.argv[1])
    mps = int(sys.argv[2])

    spark = SparkSession.builder.appName("RLSRateSourceOLS").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # OLS problem, states to be estimated are a, b and c
    # z = a*x + b * y + c + w, where w ~ N(0, 1)
    a = 0.5
    b = 0.2
    c = 1.2
    noise_param = 1
    label_expression = F.col("x") * a + F.col("y") * b + c + F.col("w")

    input_df = spark.readStream.format("rate").option("rowsPerSecond", mps).load()\
        .withColumn("mod", F.col("value") % num_states)\
        .withColumn("stateKey", F.col("mod").cast("String"))\
        .withColumn("x", (F.col("value")/num_states).cast("Integer").cast("Double"))\
        .withColumn("y", F.sqrt("x"))\
        .withColumn("bias", F.lit(1.0))\
        .withColumn("w", F.randn(0) * noise_param)\
        .withColumn("label", label_expression)

    rls = RecursiveLeastSquaresFilter(3)\
        .setStateKeyCol("stateKey")\
        .setRegularizationMatrixFactor(10E6)\
        .setForgettingFactor(0.99)

    assembler = VectorAssembler(inputCols=["x", "y", "bias"], outputCol="features")

    measurements = assembler.transform(input_df)
    query = rls.transform(measurements)\
        .writeStream\
        .queryName("RLSRateSourceOLS")\
        .outputMode("append")\
        .format("console")\
        .start()

    query.awaitTermination()