Online Linear Regression with Recursive Least Squares filter
============================================================

Recursive Least Squares (RLS) filter solves the least squares problem without requiring the complete data for training,
it can perform sequential updates to the model from a sequence of observations which is useful for streaming
applications.

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

Training multiple models is achieved by mapping samples to models. Each label and featuers can be associated with a
different model by creating a 'key' column and specifying it with `setStateKeyCol`. Not specifying any
key column will result in training a single model.

Training data is generated using streaming rate source. Streaming rate source generates
consecutive numbers with timestamps. These consecutive numbers are binned for different models and then used for
generating label & features vectors.

    .. code-block:: scala

        val numStates = 100

        // Simple linear model, states to be estimated are a, b and c
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


The estimated state distribution will be outputted in `state` struct column. The model parameters can be found at
`state.mean` field as a vector. Along with the state column, `stateKey` and `stateIndex` column can be used for
identifying different models and their incremented index.

    .. code-block:: scala

        val truncate = udf((state: DenseVector) => state.values.map(t => (math floor t * 100)/100))

        val filter = new RecursiveLeastSquaresFilter(featuresSize)
          .setStateKeyCol("stateKey")

        val query = filter.transform(features)
          .select($"stateKey", $"stateIndex", truncate($"state.mean").alias("modelParameters"))
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

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/RLSRateSourceOLS.scala>`_ for the full code

Python
------

Import RLS filter & spark, start spark session.

    .. code-block:: python

        from artan.filter import RecursiveLeastSquaresFilter
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.ml.feature import VectorAssembler

        spark = SparkSession.builder.appName("RLSExample").getOrCreate()


Each feature and label can be associated with a different model by creating a key column and specifying
it with `setStateKeyCol`. Not specifying any key column will result in training a single model.
Training data is generated using streaming rate source. Streaming rate source generates
consecutive numbers with timestamps. These consecutive numbers are binned for different models and then used for
generating label & features vectors.

    .. code-block:: python

        num_states = 10
        # Simple linear model, parameters to be estimated are a, b and c
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




The estimated state distribution will be outputted in `state` struct column. The model parameters can be found at
`state.mean` field as a vector. Along with the state column, `stateKey` and `stateIndex` column can be used for
identifying different models and their incremented index.

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
        Batch: 30
        -------------------------------------------
        +--------+----------+--------------------+
        |stateKey|stateIndex|               state|
        +--------+----------+--------------------+
        |       7|        42|[[0.4911266440390...|
        |       3|        42|[[0.4912998991072...|
        |       8|        42|[[0.4836819761355...|
        |       0|        42|[[0.5604206240212...|
        |       5|        42|[[0.5234529160112...|
        |       6|        42|[[0.5543561214337...|
        |       9|        42|[[0.4085256071251...|
        |       1|        42|[[0.4831233161778...|
        |       4|        42|[[0.5283651158175...|
        |       2|        42|[[0.4393527335453...|
        +--------+----------+--------------------+

        -------------------------------------------
        Batch: 31
        -------------------------------------------
        +--------+----------+--------------------+
        |stateKey|stateIndex|               state|
        +--------+----------+--------------------+
        |       7|        43|[[0.4949646265364...|
        |       3|        43|[[0.5051874312281...|
        |       8|        43|[[0.4697275993015...|
        |       0|        43|[[0.5407062556163...|
        |       5|        43|[[0.5223665417204...|
        |       6|        43|[[0.5438141213982...|
        |       9|        43|[[0.3951488184173...|
        |       1|        43|[[0.4639848681905...|
        |       4|        43|[[0.5232375369727...|
        |       2|        43|[[0.4618607402587...|
        +--------+----------+--------------------+

        """

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/rls_rate_source_ols.py>`_ for the full code
