# Artan
[![Build Status](https://travis-ci.com/ozancicek/artan.svg?branch=master)](https://travis-ci.com/ozancicek/artan)
[![codecov](https://codecov.io/gh/ozancicek/artan/branch/master/graph/badge.svg)](https://codecov.io/gh/ozancicek/artan)

Model-parallel bayesian filtering with Apache Spark.

- [Overview](#overview)
- [Usage](#usage)
- [Examples](#examples)

## Overview
This library provides supports for running various bayesian filters in parallel with Apache Spark. Uses arbitrary
stateful transformation capabilities of Spark DataFrames to define model-parallel bayesian filters. Therefore, it
is suitable for latent state estimation of many similar small scale systems rather than a big single system.

Both structured streaming & batch processing modes are supported. Implemented filters extend SparkML Transformers, so
you can transform a DataFrame of measurements to a DataFrame of estimated states with Kalman filters
(extended, unscented, etc,..) and various other filters as a part of your SparkML Pipeline.

Artan requires Scala 2.11, Spark 2.4+ and Python 3,6+

## Usage

In scala, filters are located at `com.ozancicek.artan.ml.filter` package.
```scala
import com.ozancicek.artan.ml.filter.LinearKalmanFilter
import org.apache.spark.ml.linalg._

val measurements: DataFrame = ... // DataFrame of measurements

// Size of the state vector
val stateSize = 2
// Size of the measurements vector
val measurementsSize = 1 //

val filter = new LinearKalmanFilter(stateSize, measurementSize)
  .setStateKeyCol("stateKey")
  .setMeasurementCol("measurement")
  .setInitialCovariance(new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
  .setProcessModel(new DenseMatrix(2, 2, Array(1.0, 0.0, 1.0, 1.0)))
  .setProcessNoise(new DenseMatrix(2, 2, Array(0.01, 0.0, 0.0, 0.01)))
  .setMeasurementNoise(new DenseMatrix(1, 1, Array(1.0)))
  .setMeasurementModel(new DenseMatrix(1, 2, Array(1.0, 0.0)))

// Transform measurements DF to state estimates
val state = filter.transform(measurements)

state.show()
// +--------+----------+--------------------+--------------------+
// |stateKey|stateIndex|               state|     stateCovariance|
// +--------+----------+--------------------+--------------------+
// |       0|         1|[-0.7208397067075...|0.999500249900037...|
// |       0|         2|[-0.4690224912583...|0.998012912493968...|
// |       0|         3|[1.67397010773909...|0.832649076225536...|
// |       0|         4|[3.44378338744333...|0.699687597788894...|
// |       0|         5|[4.46146025799578...|0.599860305854393...|
// |       1|         1|[-0.7114408860120...|0.999500249900037...|
// |       1|         2|[0.59908327905805...|0.998012912493968...|
// |       1|         3|[2.33770526334526...|0.832649076225536...|
// |       1|         4|[3.70332866441643...|0.699687597788894...|
// |       1|         5|[3.88577148836894...|0.599860305854393...|
// +--------+----------+--------------------+--------------------+
```

The supported filters in python are located at `artan.filter` package. Some of the supported filters in scala are
not yet supported in python.

```python
from artan.filter import LinearKalmanFilter
from pyspark.ml.linalg import Matrices

# DataFrame of measurements
measurements = ...
# Size of the state vector
state_size = 2

# Size of the measurements vector
measurement_size = 1

filter = LinearKalmanFilter(2, 1)\
    .setStateKeyCol("stateKey")\
    .setMeasurementCol("measurement")\
    .setInitialCovariance(Matrices.dense(2, 2, [10.0, 0.0, 0.0, 10.0]))\
    .setProcessModel(Matrices.dense(2, 2, [1.0, 0.0, 1.0, 1.0]))\
    .setProcessNoise(Matrices.dense(2, 2, [0.01, 0.0, 0.0, 0.01]))\
    .setMeasurementNoise(Matrices.dense(1, 1, [1.0]))\
    .setMeasurementModel(Matrices.dense(1, 2, [1.0, 0.0]))

# Transform measurements DF to state estimates
state = filter.transform(measurements)

state.show()
# +--------+----------+--------------------+--------------------+
# |stateKey|stateIndex|               state|     stateCovariance|
# +--------+----------+--------------------+--------------------+
# |       0|         1|[-0.9837995930710...|0.999950002500125...|
# |       0|         2|[-0.8434180999763...|0.999800129920049...|
# |       0|         3|[3.13016426536432...|0.833272243145062...|
# |       1|         1|[0.77903611655215...|0.999950002500125...|
# |       1|         2|[-1.7712651068330...|0.999800129920049...|
# |       1|         3|[1.89342894288159...|0.833272243145062...|
# +--------+----------+--------------------+--------------------+

```

## Examples

See [examples](examples/src/main) for all sample scripts.

### Streaming examples
- Local linear trend filtering with Linear Kalman Filter ([python](examples/src/main/python/streaming/lkf_rate_source_llt.py), [scala](examples/src/main/scala/com/ozancicek/artan/examples/streaming/LKFRateSourceLLT.scala))
- Recursive least squares ([scala](examples/src/main/scala/com/ozancicek/artan/examples/streaming/RLSRateSourceOLS.scala))
- GLM estimation with Extended Kalman Filter, gaussian noise & log link ([scala](examples/src/main/scala/com/ozancicek/artan/examples/streaming/EKFRateSourceGLMLog.scala))
