Online Linear Regression with Kalman Filter
===========================================

Recursive estimation of least squares can be easily done with a Kalman Filter. Using state-space
representation, the following linear model:


    .. image:: https://latex.codecogs.com/svg.latex?%5C%5C%20Y_t%20%3D%20%5Cbeta%20X_t%20&plus;%20%5Cepsilon%20%3A%20%5Cepsilon%20%24%5Csim%24%20N%280%2C%20R%29%20%5Cquad%20t%3D%201%2C%202%2C%20...%20T%20%5C%5C

Can be represented in state-space form by:

    .. image:: https://latex.codecogs.com/svg.latex?%5C%5C%20V_t%20%3D%20A_t%20V_%7Bt%20-%201%7D%20&plus;%20q_%7Bt%7D%3A%20q_t%20%24%5Csim%24%20N%280%2C%20Q%29%20%5Cquad%20%28state%20%5C%20process%20%5C%20equation%29%20%5C%5C%20Z_t%20%3D%20H_t%20V_t%20&plus;%20r_t%3A%20r_t%20%24%5Csim%24%20N%280%2C%20R%29%20%5Cquad%20%28measurement%20%5C%20equation%29%20%5C%5C%20%5C%5C%20A_t%20%3D%20I%5C%5C%20H_t%20%3D%20X_t%5C%5C%20q_t%20%3D%200

At each time step `t`, the state would give an estimate of the model parameters.

Scala
-----
Import Kalman filter and start spark session.

    .. code-block:: scala

        import com.github.ozancicek.artan.ml.filter.LinearKalmanFilter
        import org.apache.spark.ml.linalg._
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._

        val spark = SparkSession
          .builder
          .appName("LKFRateSourceOLS")
          .getOrCreate

        import spark.implicits._
        val rowsPerSecond = 10
        val numStates = 10


Define the model parameters and udf's to generate training data.

Since the aim is to estimate the model parameters,
the state of the filter is model parameters. Label will become measurements vector with size 1. Features
will be denoted with a 1x3 measurement model matrix, which will map the state to measurements with a dot product.

    .. code-block:: scala

        // OLS problem, states to be estimated are a, b and c
        // z = a*x + b * y + c + w, where w ~ N(0, 1)

        val a = 0.5
        val b = 0.2
        val c = 1.2
        val stateSize = 3
        val measurementSize = 1
        val noiseParam = 1.0

        val featuresUDF = udf((x: Double, y: Double) => {
          new DenseMatrix(measurementSize, stateSize, Array(x, y, 1.0))
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

Initialize the filter & run the query with console sink.

All of the filter parameters can be set either as an input dataframe column, or directly the value itself with
`ml.linalg.Vector` or `ml.linalg.Matrix`. Specifying parameters from dataframe columns will allow you to have
varying values across measurements/filters.

For this example, measurement and measurement model should be varying across
measurements, so they're set from dataframe columns. Process model, process noise, measurement noise and initial covariance
can be same for all measurements/filters, so their values are set directly with matrices.

    .. code-block:: scala

        val filter = new LinearKalmanFilter(stateSize, measurementSize)
          .setInitialCovariance(
            new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
          .setStateKeyCol("stateKey")
          .setMeasurementCol("label")
          .setMeasurementModelCol("features")
          .setProcessModel(DenseMatrix.eye(stateSize))
          .setProcessNoise(DenseMatrix.zeros(stateSize, stateSize))
          .setMeasurementNoise(DenseMatrix.eye(measurementSize))

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


See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/ozancicek/artan/examples/streaming/LKFRateSourceOLS.scala>`_ for the full code


Python
------

Import Kalman Filter and start spark session.

    .. code-block:: python

    from artan.filter import LinearKalmanFilter

    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.ml.linalg import Matrices, Vectors, MatrixUDT, VectorUDT
    from pyspark.sql.types import StringType

    spark = SparkSession.builder.appName("LKFRateSourceOLS").getOrCreate()

    num_states = 10
    measurements_per_sec = 10


Define model parameters, #models and udf's to generate training data.

Since the aim is to estimate the model parameters,
the state of the filter is model parameters. Label will become measurements vector with size 1. Features
will be denoted with a 1x3 measurement model matrix, which will map the state to measurements with a dot product.

    .. code-block:: python

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

        features = spark.readStream.format("rate").option("rowsPerSecond", measurements_per_sec).load()\
            .withColumn("mod", F.col("value") % num_states)\
            .withColumn("stateKey", F.col("mod").cast("String"))\
            .withColumn("x", (F.col("value")/num_states).cast("Integer").cast("Double"))\
            .withColumn("y", F.sqrt("x"))\
            .withColumn("w", F.randn(0) * noise_param)\
            .withColumn("label", label_udf("x", "y", "w"))\
            .withColumn("features", features_udf("x", "y"))

Initialize the filter & run the query with console sink.

All of the filter parameters can be set either as an input dataframe column, or directly the value itself with
`ml.linalg.Vector` or `ml.linalg.Matrix`. Specifying parameters from dataframe columns will allow you to have
varying values across measurements/filters.

For this example, measurement and measurement model should be varying across
measurements, so they're set from dataframe columns. Process model, process noise, measurement noise and initial covariance
can be same for all measurements/filters, so their values are set directly with matrices.

    .. code-block:: python

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

        """
        -------------------------------------------
        Batch: 32
        -------------------------------------------
        +--------+----------+-------------------+
        |stateKey|stateIndex|    modelParameters|
        +--------+----------+-------------------+
        |       7|        74|[0.55, -0.30, 2.29]|
        |       3|        74|[0.55, -0.26, 1.87]|
        |       8|        74| [0.51, 0.18, 1.14]|
        |       0|        74| [0.47, 0.52, 0.41]|
        |       5|        74|[0.52, -0.01, 1.70]|
        |       6|        74| [0.49, 0.32, 1.13]|
        |       9|        74| [0.49, 0.39, 0.68]|
        |       1|        74|[0.52, -0.09, 2.15]|
        |       4|        74| [0.50, 0.05, 2.13]|
        |       2|        74| [0.49, 0.34, 0.77]|
        +--------+----------+-------------------+

        -------------------------------------------
        Batch: 33
        -------------------------------------------
        +--------+----------+-------------------+
        |stateKey|stateIndex|    modelParameters|
        +--------+----------+-------------------+
        |       7|        75|[0.54, -0.19, 2.11]|
        |       7|        76|[0.54, -0.22, 2.16]|
        |       3|        75|[0.55, -0.24, 1.84]|
        |       3|        76|[0.55, -0.23, 1.82]|
        |       8|        75| [0.50, 0.18, 1.13]|
        |       8|        76| [0.50, 0.21, 1.10]|
        |       0|        75| [0.47, 0.54, 0.38]|
        |       0|        76| [0.47, 0.54, 0.38]|
        |       5|        75| [0.51, 0.07, 1.58]|
        |       5|        76| [0.50, 0.13, 1.50]|
        |       6|        75| [0.48, 0.35, 1.07]|
        |       6|        76| [0.48, 0.35, 1.07]|
        |       9|        75| [0.49, 0.35, 0.74]|
        |       9|        76| [0.49, 0.37, 0.71]|
        |       1|        75|[0.51, -0.03, 2.07]|
        |       1|        76|[0.51, -0.02, 2.04]|
        |       4|        75| [0.50, 0.06, 2.12]|
        |       4|        76| [0.50, 0.04, 2.15]|
        |       2|        75| [0.49, 0.36, 0.75]|
        |       2|        76| [0.49, 0.33, 0.79]|
        +--------+----------+-------------------+

        """


See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/lkf_rate_source_ols.py>`_ for the full code
