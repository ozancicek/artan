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
    
    `spark-submit --py-files artan.zip --jars artan-examples-assembly-VERSION.jar rls_rate_source_ols.py 2 2`
    -------------------------------------------
    Batch: 1
    -------------------------------------------
    +--------+----------+--------------------+--------------------+
    |stateKey|stateIndex|               state|          covariance|
    +--------+----------+--------------------+--------------------+
    |       0|         1|[0.0,0.0,2.060506...|1.01010101010101E...|
    |       0|         2|[0.52736332451361...|3401013.725368142...|
    |       0|         3|[4.02137556053600...|9.282205848464974...|
    |       0|         4|[1.25538976438005...|2.678321189223935...|
    |       1|         1|[0.0,0.0,0.872789...|1.01010101010101E...|
    |       1|         2|[0.53765773789408...|3401013.725368142...|
    |       1|         3|[1.56606579234287...|9.282205848464974...|
    |       1|         4|[1.73082635494717...|2.678321189223935...|
    +--------+----------+--------------------+--------------------+
    
    -------------------------------------------
    Batch: 2
    -------------------------------------------
    +--------+----------+--------------------+--------------------+
    |stateKey|stateIndex|               state|          covariance|
    +--------+----------+--------------------+--------------------+
    |       0|         5|[0.92953864906319...|1.21265315495273 ...|
    |       0|         6|[1.11491403834006...|0.441497166386165...|
    |       0|         7|[1.06936981333432...|0.423316873905838...|
    |       1|         5|[0.74633860822657...|1.21265315495273 ...|
    |       1|         6|[0.05539862379861...|0.441497166386165...|
    |       1|         7|[-0.0746118638759...|0.423316873905838...|
    +--------+----------+--------------------+--------------------+
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