# Documentation

## Quick Start

This tutorial will show you how to setup a very simple spark application with this library. The example consists of
online training of a model-parallel Recursive Least Squares with spark structured streaming.

## Online Recursive Least Squares
 
As its name suggests, Recursive Least Squares (RLS) is a recursive solution to the least squares problem. RLS
does not require the complete data for training, it can perform sequential updates to the model from a
sequence of observations which is useful for streaming applications.

### Prerequisites
 
Install Spark 2.4+, Scala 2.11 and Python 3.6+. Spark shell or pyspark shell can be run with maven coordinates
using ``--packages`` argument. This will place all required jars and python files to appropriate executor and driver
paths.

    spark-shell --packages com.github.ozancicek:artan_2.11:0.1.0
    pyspark --packages com.github.ozancicek:artan_2.11:0.1.0
    spark-submit --packages com.github.ozancicek:artan_2.11:0.1.0


For developing with Scala, the dependencies can be retrieved from Maven Central.

    libraryDependencies += "com.github.ozancicek" %% "artan" % "0.1.0"
    
For developing with Python, the dependencies can be installed with pip.

    pip install artan
    
Note that pip will only install the python dependencies, which is not enough to submit jobs to spark cluster. 
To submit pyspark jobs, `--packages='com.github.ozancicek:artan_2.11:0.1.0'` argument should still be specified in
order to download necessary jars from maven central.
 
 
#### Scala
 
Import RLS filter & spark, start spark session.
 
```scala
import com.github.ozancicek.artan.ml.filter.RecursiveLeastSquaresFilter
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._


val spark = SparkSession
  .builder
  .appName("RLSExample")
  .getOrCreate
 
import spark.implicits._
``` 

Define model parameters, #models and udf's to generate training data.

```scala
val numStates = 100

// OLS problem, states to be estimated are a, b and c
// z = a*x + b * y + c + w, where w ~ N(0, 1)

val a = 0.5
val b = 0.2
val c = 1.2
val noiseParam = 1.0

val featuresUDF = udf((x: Double, y: Double) => {
  new DenseVector(Array(x, y, 1.0))
})

val labelUDF = udf((x: Double, y: Double, w: Double) => {
  a*x + b*y + c + w
})
```

Generate the training data using streaming rate source. 

```scala

val features = spark.readStream.format("rate")
  .option("rowsPerSecond", rowsPerSecond)
  .load()
  .withColumn("mod", $"value" % numStates)
  .withColumn("stateKey", $"mod".cast("String"))
  .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
  .withColumn("y", sqrt($"x"))
  .withColumn("label", labelUDF($"x", $"y", randn() * noiseParam))
  .withColumn("features", featuresUDF($"x", $"y"))
``` 

Initialize the filter & run the query with console sink

```scala
val truncate = udf((state: DenseVector) => state.values.map(t => (math floor t * 100)/100))

val filter = new RecursiveLeastSquaresFilter(3)
  .setStateKeyCol("stateKey")
  
val query = filter.transform(features)
  .select($"stateKey", $"stateIndex", truncate($"state").alias("modelParameters"))
  .writeStream
  .queryName("RLSRateSourceOLS")
  .outputMode("append")
  .format("console")
  .start()

query.awaitTermination()
```

#### Python

Import RLS filter & spark, start spark session.

```python

from artan.filter import RecursiveLeastSquaresFilter

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler


spark = SparkSession.builder.appName("RLSExample").getOrCreate()
```

Define model parameters, #models and expressions to generate training data.

```python
num_states = 10
# OLS problem, states to be estimated are a, b and c
# z = a*x + b * y + c + w, where w ~ N(0, 1)
a = 0.5
b = 0.2
c = 1.2
noise_param = 1
label_expression = F.col("x") * a + F.col("y") * b + c + F.col("w")
```

Generate the training data using streaming rate source. 

```python
input_df = spark.readStream.format("rate").option("rowsPerSecond", mps).load()\
    .withColumn("mod", F.col("value") % num_states)\
    .withColumn("stateKey", F.col("mod").cast("String"))\
    .withColumn("x", (F.col("value")/num_states).cast("Integer").cast("Double"))\
    .withColumn("y", F.sqrt("x"))\
    .withColumn("bias", F.lit(1.0))\
    .withColumn("w", F.randn(0) * noise_param)\
    .withColumn("label", label_expression)
    
assembler = VectorAssembler(inputCols=["x", "y", "bias"], outputCol="features")
 
measurements = assembler.transform(input_df)
```

Initialize the filter & run the query with console sink

```python
rls = RecursiveLeastSquaresFilter(3)\
    .setStateKeyCol("stateKey")

query = rls.transform(measurements)\
    .writeStream\
    .queryName("RLSRateSourceOLS")\
    .outputMode("append")\
    .format("console")\
    .start()

query.awaitTermination()
```