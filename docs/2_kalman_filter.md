# Documentation


This tutorial will show you a streaming Kalman filter example with this library. The example consists of
online training of a model-parallel Recursive Least Squares, using a Kalman filter with
spark structured streaming.


## Online Recursive Least Squares with Kalman filter

Recursive estimation of least squares can be easily done with a Kalman filter. Using state-space
representation, the following linear model:


![linmod](https://latex.codecogs.com/svg.latex?%5C%5C%20Y_t%20%3D%20%5Cbeta%20X_t%20&plus;%20%5Cepsilon%20%3A%20%5Cepsilon%20%24%5Csim%24%20N%280%2C%20R%29%20%5Cquad%20t%3D%201%2C%202%2C%20...%20T%20%5C%5C)

Can be represented in state-space form by:

![statespace](https://latex.codecogs.com/svg.latex?%5C%5C%20V_t%20%3D%20A_t%20V_%7Bt%20-%201%7D%20&plus;%20q_%7Bt%7D%3A%20q_t%20%24%5Csim%24%20N%280%2C%20Q%29%20%5Cquad%20%28state%20%5C%20process%20%5C%20equation%29%20%5C%5C%20Z_t%20%3D%20H_t%20V_t%20&plus;%20r_t%3A%20r_t%20%24%5Csim%24%20N%280%2C%20R%29%20%5Cquad%20%28measurement%20%5C%20equation%29%20%5C%5C%20%5C%5C%20A_t%20%3D%20I%5C%5C%20H_t%20%3D%20X_t%5C%5C%20q_t%20%3D%200)

At each time step `t`, the state would give an estimate of the model parameters.

#### Scala

```scala
import com.github.ozancicek.artan.ml.filter.LinearKalmanFilter
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession
  .builder
  .appName("LKFRateSourceOLS")
  .getOrCreate

import spark.implicits._
```

```scala
// OLS problem, states to be estimated are a, b and c
// z = a*x + b * y + c + w, where w ~ N(0, 1)

val a = 0.5
val b = 0.2
val c = 1.2
val stateSize = 3
val measurementsSize = 1
val noiseParam = 1.0

val featuresUDF = udf((x: Double, y: Double) => {
  new DenseMatrix(measurementsSize, stateSize, Array(x, y, 1.0))
})

val labelUDF = udf((x: Double, y: Double, r: Double) => {
  new DenseVector(Array(a*x + b*y + c + r))
})

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

```scala
val filter = new LinearKalmanFilter(stateSize, measurementsSize)
  .setInitialCovariance(
    new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
  .setStateKeyCol("stateKey")
  .setMeasurementCol("label")
  .setMeasurementModelCol("features")
  .setProcessModel(DenseMatrix.eye(stateSize))
  .setProcessNoise(DenseMatrix.zeros(stateSize, stateSize))
  .setMeasurementNoise(DenseMatrix.eye(measurementsSize))

val truncate = udf((state: DenseVector) => state.values.map(t => (math floor t * 100)/100))

val query = filter.transform(features)
  .select($"stateKey", $"stateIndex", truncate($"state").alias("modelParameters"))
  .writeStream
  .queryName("LKFRateSourceOLS")
  .outputMode("append")
  .format("console")
  .start()

query.awaitTermination()

/*
-------------------------------------------
Batch: 53
-------------------------------------------
+--------+----------+-------------------+
|stateKey|stateIndex|    modelParameters|
+--------+----------+-------------------+
|       7|        61| [0.47, 0.48, 0.28]|
|       3|        61| [0.46, 0.55, 0.56]|
|       8|        61| [0.45, 0.61, 0.22]|
|       0|        61|[0.53, -0.14, 1.81]|
|       5|        61| [0.49, 0.27, 1.01]|
|       6|        61| [0.47, 0.35, 1.02]|
|       9|        61|[0.52, -0.13, 1.95]|
|       1|        61|  [0.52, 0.0, 1.63]|
|       4|        61| [0.51, 0.13, 1.22]|
|       2|        61|[0.53, -0.19, 1.82]|
+--------+----------+-------------------+

-------------------------------------------
Batch: 54
-------------------------------------------
+--------+----------+-------------------+
|stateKey|stateIndex|    modelParameters|
+--------+----------+-------------------+
|       7|        62| [0.47, 0.49, 0.27]|
|       3|        62| [0.46, 0.54, 0.57]|
|       8|        62| [0.45, 0.65, 0.17]|
|       0|        62| [0.53, -0.1, 1.76]|
|       5|        62| [0.49, 0.27, 1.01]|
|       6|        62| [0.48, 0.32, 1.06]|
|       9|        62|[0.52, -0.11, 1.93]|
|       1|        62| [0.51, 0.06, 1.56]|
|       4|        62| [0.52, 0.06, 1.31]|
|       2|        62| [0.54, -0.24, 1.9]|
+--------+----------+-------------------+
*/

```

See [examples](/examples/src/main/scala/com/ozancicek/artan/examples/streaming/LKFRateSourceOLS.scala) for the full code