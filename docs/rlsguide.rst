Online Linear Regression with Recursive Least Squares filter
============================================================

As its name suggests, Recursive Least Squares (RLS) is a recursive solution to the least squares problem. RLS
does not require the complete data for training, it can perform sequential updates to the model from a
sequence of observations which is useful for streaming applications.

Scala
-----

Import RLS filter & spark, start spark session.

    .. code-block:: scala

        import com.github.ozancicek.artan.ml.filter.RecursiveLeastSquaresFilter
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._
        import org.apache.spark.ml.linalg._


        val spark = SparkSession
          .builder
          .appName("RLSExample")
          .getOrCreate

        import spark.implicits._

Define model parameters, #models and udf's to generate training data.

Each feature and label can be associated with a
different model by creating a key column & specifying it with `setStateKeyCol`. Not specifying any key column will result
in training a single model. Training data is generated using streaming rate source. Streaming rate source generates
consecutive numbers with timestamps. These consecutive numbers are binned for different models and then used for
generating label & features vectors.

    .. code-block:: scala

        val numStates = 100

        // OLS problem, states to be estimated are a, b and c
        // z = a*x + b*y + c + w, where w ~ N(0, 1)

        val a = 0.5
        val b = 0.2
        val c = 1.2
        val noiseParam = 1.0
        val featuresSize = 3

        val featuresUDF = udf((x: Double, y: Double) => {
            new DenseVector(Array(x, y, 1.0))
        })

        val labelUDF = udf((x: Double, y: Double, w: Double) => {
            a*x + b*y + c + w
        })

        val features = spark.readStream.format("rate")
          .option("rowsPerSecond", 10)
          .load()
          .withColumn("mod", $"value" % numStates)
          .withColumn("stateKey", $"mod".cast("String"))
          .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
          .withColumn("y", sqrt($"x"))
          .withColumn("label", labelUDF($"x", $"y", randn() * noiseParam))
          .withColumn("features", featuresUDF($"x", $"y"))


Initialize the filter & run the query with console sink.

Trained model parameters are located at `state`
column as a vector. Along with the state column, `stateKey` and `stateIndex` column can be used for indentifying
different models and their incremented index.

    .. code-block:: scala

        val truncate = udf((state: DenseVector) => state.values.map(t => (math floor t * 100)/100))

        val filter = new RecursiveLeastSquaresFilter(featuresSize)
          .setStateKeyCol("stateKey")

        val query = filter.transform(features)
          .select($"stateKey", $"stateIndex", truncate($"state").alias("modelParameters"))
          .writeStream
          .queryName("RLSRateSourceOLS")
          .outputMode("append")
          .format("console")
          .start()

        query.awaitTermination()

        /*
        Batch: 65
        -------------------------------------------
        +--------+----------+-------------------+
        |stateKey|stateIndex|    modelParameters|
        +--------+----------+-------------------+
        |       7|        68|[0.54, -0.19, 1.98]|
        |       3|        68|  [0.5, 0.11, 1.41]|
        |       8|        68|[0.53, -0.13, 1.89]|
        |       0|        68| [0.46, 0.53, 0.34]|
        |       5|        68|   [0.5, 0.2, 1.05]|
        |       6|        68| [0.45, 0.68, 0.18]|
        |       9|        68|[0.53, -0.15, 1.82]|
        |       1|        68|  [0.5, 0.09, 2.17]|
        |       4|        68| [0.51, 0.11, 1.17]|
        |       2|        68|  [0.48, 0.35, 0.9]|
        +--------+----------+-------------------+

        -------------------------------------------
        Batch: 66
        -------------------------------------------
        +--------+----------+-------------------+
        |stateKey|stateIndex|    modelParameters|
        +--------+----------+-------------------+
        |       7|        69|[0.54, -0.18, 1.96]|
        |       3|        69| [0.49, 0.19, 1.28]|
        |       8|        69|[0.53, -0.19, 1.99]|
        |       0|        69|  [0.45, 0.6, 0.23]|
        |       5|        69| [0.51, 0.14, 1.15]|
        |       6|        69| [0.45, 0.71, 0.14]|
        |       9|        69| [0.53, -0.1, 1.75]|
        |       1|        69| [0.49, 0.15, 2.09]|
        |       4|        69|  [0.51, 0.1, 1.18]|
        |       2|        69| [0.49, 0.25, 1.04]|
        +--------+----------+-------------------+
        */

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/ozancicek/artan/examples/streaming/RLSRateSourceOLS.scala>`_ for the full code

Python
------

Import RLS filter & spark, start spark session.

    .. code-block:: python

        from artan.filter import RecursiveLeastSquaresFilter
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.ml.feature import VectorAssembler

        spark = SparkSession.builder.appName("RLSExample").getOrCreate()


Define model parameters, #models and expressions to generate training data.

Each feature and label can be associated with a
different model by creating a key column & specifying it with `setStateKeyCol`. Not specifying any key column will result
in training a single model. Training data is generated using streaming rate source. Streaming rate source generates
consecutive numbers with timestamps. These consecutive numbers are binned for different models and then used for
generating label & features vectors.

    .. code-block:: python

        num_states = 10
        # OLS problem, states to be estimated are a, b and c
        # z = a*x + b*y + c + w, where w ~ N(0, 1)
        a = 0.5
        b = 0.2
        c = 1.2
        noise_param = 1
        features_size = 3
        label_expression = F.col("x") * a + F.col("y") * b + c + F.col("w")

        input_df = spark.readStream.format("rate").option("rowsPerSecond", 10).load()\
            .withColumn("mod", F.col("value") % num_states)\
            .withColumn("stateKey", F.col("mod").cast("String"))\
            .withColumn("x", (F.col("value")/num_states).cast("Integer").cast("Double"))\
            .withColumn("y", F.sqrt("x"))\
            .withColumn("bias", F.lit(1.0))\
            .withColumn("w", F.randn(0) * noise_param)\
            .withColumn("label", label_expression)

        assembler = VectorAssembler(inputCols=["x", "y", "bias"], outputCol="features")

        measurements = assembler.transform(input_df)



Initialize the filter & run the query with console sink.

Trained model parameters are located at `state`
column as a vector. Along with the state column, `stateKey` and `stateIndex` column can be used for indentifying
different models and their incremented index.

    .. code-block:: python

        rls = RecursiveLeastSquaresFilter(features_size)\
            .setStateKeyCol("stateKey")

        query = rls.transform(measurements)\
            .writeStream\
            .queryName("RLSRateSourceOLS")\
            .outputMode("append")\
            .format("console")\
            .start()

        query.awaitTermination()

        """
        -------------------------------------------
        Batch: 36
        -------------------------------------------
        +--------+----------+--------------------+--------------------+
        |stateKey|stateIndex|               state|          covariance|
        +--------+----------+--------------------+--------------------+
        |       7|        50|[0.52522671801750...|0.002525312259059...|
        |       3|        50|[0.45416326001988...|0.002525312259059...|
        |       8|        50|[0.43784192991338...|0.002525312259059...|
        |       0|        50|[0.51435805075613...|0.002525312259059...|
        |       5|        50|[0.54943787474521...|0.002525312259059...|
        |       6|        50|[0.45201596104561...|0.002525312259059...|
        |       9|        50|[0.46456128079570...|0.002525312259059...|
        |       1|        50|[0.44471842109727...|0.002525312259059...|
        |       4|        50|[0.51927827156396...|0.002525312259059...|
        |       2|        50|[0.47024488052215...|0.002525312259059...|
        +--------+----------+--------------------+--------------------+

        -------------------------------------------
        Batch: 37
        -------------------------------------------
        +--------+----------+--------------------+--------------------+
        |stateKey|stateIndex|               state|          covariance|
        +--------+----------+--------------------+--------------------+
        |       7|        51|[0.52416295086994...|0.002405612639984...|
        |       3|        51|[0.44793632024707...|0.002405612639984...|
        |       8|        51|[0.45147440917940...|0.002405612639984...|
        |       0|        51|[0.50187121102737...|0.002405612639984...|
        |       5|        51|[0.55364576956303...|0.002405612639984...|
        |       6|        51|[0.47217482082352...|0.002405612639984...|
        |       9|        51|[0.46444553756938...|0.002405612639984...|
        |       1|        51|[0.45289693949378...|0.002405612639984...|
        |       4|        51|[0.51771140555410...|0.002405612639984...|
        |       2|        51|[0.46263280865422...|0.002405612639984...|
        +--------+----------+--------------------+--------------------+
        """

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/rls_rate_source_ols.py>`_ for the full code
