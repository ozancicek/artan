Multiple-Model Adaptive Estimation
==================================

The example demonstrated here is same with the one in :ref:`Kalman Filter section <Online Linear Regression with Kalman Filter>`


Import Kalman filter and start spark session.

    .. code-block:: scala

        import com.github.ozancicek.artan.ml.filter.LinearKalmanFilter
        import org.apache.spark.ml.linalg._
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._

        val spark = SparkSession
          .builder
          .appName("MMAERateSourceOLS")
          .getOrCreate

        import spark.implicits._

        val numStates = 10
        val rowsPerSecond = 10

Define the model parameters and udf's to generate training data.

    .. code-block:: scala

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

        val filter = new LinearKalmanFilter(stateSize, measurementsSize)
          .setInitialCovariance(
            new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
          .setStateKeyCol("stateKey")
          .setMeasurementCol("label")
          .setMeasurementModelCol("features")
          .setProcessModel(DenseMatrix.eye(stateSize))
          .setProcessNoise(DenseMatrix.zeros(stateSize, stateSize))
          .setMeasurementNoise(DenseMatrix.eye(measurementsSize))
          .setSlidingLikelihoodWindow(10)
          .setEventTimeCol("timestamp")
          .setWatermarkDuration("2 seconds")
          .setEnableMultipleModelAdaptiveEstimation
          .setMultipleModelMeasurementWindowDuration("5 seconds")


Generate the data & run the query with console sink.

    .. code-block:: scala

        val features = spark.readStream.format("rate")
          .option("rowsPerSecond", rowsPerSecond)
          .load()
          .withColumn("mod", $"value" % numStates)
          .withColumn("stateKey", $"mod".cast("String"))
          .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
          .withColumn("y", sqrt($"x"))
          .withColumn("label", labelUDF($"x", $"y", randn() * noiseParam))
          .withColumn("features", featuresUDF($"x", $"y"))

        val truncate = udf((state: DenseVector) => state.values.map(t => (math floor t * 100)/100))

        val query = filter.transform(features)
          .select( $"stateIndex", truncate($"state").alias("modelParameters"), $"timestamp")
          .writeStream
          .queryName("MMAERateSourceOLS")
          .outputMode("append")
          .format("console")
          .start()

        query.awaitTermination()

        /*
        -------------------------------------------
        Batch: 49
        -------------------------------------------
        +----------+------------------+--------------------+
        |stateIndex|   modelParameters|           timestamp|
        +----------+------------------+--------------------+
        |        94|[0.49, 0.24, 1.01]|[2020-04-11 18:48...|
        |        93|[0.49, 0.24, 1.03]|[2020-04-11 18:48...|
        |        91| [0.5, 0.17, 1.23]|[2020-04-11 18:48...|
        |        95| [0.5, 0.15, 1.36]|[2020-04-11 18:48...|
        |        92| [0.5, 0.17, 1.25]|[2020-04-11 18:48...|
        +----------+------------------+--------------------+
        -------------------------------------------
        Batch: 52
        -------------------------------------------
        +----------+------------------+--------------------+
        |stateIndex|   modelParameters|           timestamp|
        +----------+------------------+--------------------+
        |        98|  [0.5, 0.12, 1.5]|[2020-04-11 18:48...|
        |        99|[0.49, 0.21, 1.16]|[2020-04-11 18:48...|
        |        96|[0.51, 0.03, 1.54]|[2020-04-11 18:48...|
        |       100|  [0.5, 0.13, 1.5]|[2020-04-11 18:48...|
        |        97| [0.5, 0.08, 1.57]|[2020-04-11 18:48...|
        +----------+------------------+--------------------+
        */

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/MMAERateSourceOLS.scala>`_ for the full code