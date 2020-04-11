Online Nonlinear Regression with Unscented Kalman Filter
=======================================================

Similar with EKF, Unscented Kalman Filter (UKF) can be used for systems where measurement or state process updates
are nonlinear functions. The advantage of UKF over EKF is that, For UKF you don't have to specify jacobian function of the
nonlinear update. UKF uses deterministic sampling algorithms to estimate state and its covariance, so instead you have
to specify sampling algorithm and its hyperparameters that suits your problem. The example demonstrated here is same with
the :ref:`previous section <Online Nonlinear Regression with Extended Kalman Filter>`


Import UKF and start spark session.

    .. code-block:: scala

        import com.github.ozancicek.artan.ml.filter.UnscentedKalmanFilter
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._
        import org.apache.spark.ml.linalg._

        val rowsPerSecond = 10
        val numStates = 10


Define the model parameters and udf's to generate training data.

For UKF, we only need to define the nonlinear transformation. Similar with EKF, the signature of this function
should be ``(Vector, Matrix) => Vector``.

    .. code-block:: scala

        // GLM with log link, states to be estimated are a, b
        // y = exp(a*x + b) + w, where w ~ N(0, 1)
        val a = 0.2
        val b = 0.7
        val noiseParam = 1.0
        val stateSize = 2
        val measurementSize = 1

        // UDF's for generating measurement vector ([y]) and measurement model matrix ([[x ,1]])
        val measurementUDF = udf((x: Double, r: Double) => {
          val measurement = scala.math.exp(a * x + b) + r
          new DenseVector(Array(measurement))
        })

        val measurementModelUDF = udf((x: Double) => {
          new DenseMatrix(1, 2, Array(x, 1.0))
        })

        // No jac func is needed compared to EKF
        val measurementFunc = (in: Vector, model: Matrix) => {
          val measurement = model.multiply(in)
          measurement.values(0) = scala.math.exp(measurement.values(0))
          measurement
        }

        val filter = new UnscentedKalmanFilter(stateSize, measurementSize)
          .setStateKeyCol("stateKey")
          .setInitialCovariance(
            DenseMatrix.eye(2))
          .setMeasurementCol("measurement")
          .setMeasurementModelCol("measurementModel")
          .setProcessModel(DenseMatrix.eye(2))
          .setProcessNoise(DenseMatrix.zeros(2, 2))
          .setMeasurementNoise(DenseMatrix.eye(1))
          .setMeasurementFunction(measurementFunc)
          .setCalculateMahalanobis

Generate the data & run the query with console sink.

    .. code-block:: scala

        val measurements = spark.readStream.format("rate")
          .option("rowsPerSecond", rowsPerSecond)
          .load()
          .withColumn("mod", $"value" % numStates)
          .withColumn("stateKey", $"mod".cast("String"))
          .withColumn("x", ($"value"/numStates).cast("Integer").cast("Double"))
          .withColumn("measurement", measurementUDF($"x", randn() * noiseParam))
          .withColumn("measurementModel", measurementModelUDF($"x"))

        val query = filter.transform(measurements)
          .writeStream
          .queryName("UKFRateSourceGLMLog")
          .outputMode("append")
          .format("console")
          .start()

        query.awaitTermination()
        /*
        -------------------------------------------
        Batch: 43
        -------------------------------------------
        +--------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
        |stateKey|stateIndex|               state|     stateCovariance|            residual|  residualCovariance|        mahalanobis|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
        |       7|        51|[0.19999855713990...|2.717895323579557...|[-0.2254410591267...| 2.225820190586552  |0.15110818086092595|
        |       3|        51|[0.20000549084503...|2.717150686585378...|[-0.5654512399705...| 2.225677602511927  | 0.3790216632241064|
        |       8|        51|[0.19999143923324...|2.716558504037461...|[-0.1595262547853...| 2.225473327382053  | 0.1069352705506542|
        |       0|        51|[0.19999972916295...|2.717199913329036...|[1.0167240073496941]| 2.225552393246534  | 0.6815287279852043|
        |       5|        51|[0.19999757539110...|2.729421815651810...|[0.19505260117148...|2.2295271297379067  | 0.1306307335569149|
        |       6|        51|[0.19999012713869...|2.71838117643021E...|[0.3785885690158466]|2.2259977910943984  |0.25374946329990344|
        |       9|        51|[0.20000137689024...|2.719910029136810...|[-1.8068528499861...|2.2264190575400065  | 1.2109308113752884|
        |       1|        51|[0.19999292852721...|2.717676139424999...|[1.5988637913396815]| 2.225784202000507  |  1.071691878585769|
        |       4|        51|[0.20000395872207...|2.718046923741906...|[-1.5466027889633...|2.2259155359167364  | 1.0366316124322652|
        |       2|        51|[0.20000971110099...|2.717475974222968...|[-1.5416258407494...|2.2258200870745464  | 1.0333178999554362|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+

        -------------------------------------------
        Batch: 44
        -------------------------------------------
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        |stateKey|stateIndex|               state|     stateCovariance|            residual|  residualCovariance|         mahalanobis|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        |       7|        52|[0.20000207032942...|1.8216962246316E-...| [1.750796229171101]|2.2257303589310857  |  1.1735438659779236|
        |       3|        52|[0.20000160104336...|1.821252819219506...|[-1.9387669152856...|2.2256393255561684  |  1.2995655996042372|
        |       8|        52|[0.19999154737300...|1.821056388672728...|[0.05390878603066...|2.2254351972195106  |0.036136998344480975|
        |       0|        52|[0.20000399573738...|1.82133819999608E...|[2.1265805893344805]|2.2255943807416214  |  1.4254724803988705|
        |       5|        52|[0.19999798049981...|1.827222463347535...|[0.20133526511926...|2.2284302446429005  |  0.1348715438576178|
        |       6|        52|[0.19999220783719...|1.821860403967139...| [1.036761540970474]| 2.225870052043158  |   0.694910662428129|
        |       9|        52|[0.20000104781302...|1.822708243581388...|[-0.1639191357098...| 2.226146770938694  | 0.10986332997430581|
        |       1|        52|[0.19999438603204...|1.821567553078202...|[0.7263834670957294]|2.2257302838857655  |  0.4868886866020558|
        |       4|        52|[0.20000434979213...|1.821783707325596...|[0.19488410425401...|2.2257648672162538  | 0.13062814985752372|
        |       2|        52|[0.20000407611972...|1.821467582002353...|[-2.808469120922382]|2.2256989968567216  |   1.882506299396078|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        */

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/UKFRateSourceGLMLog.scala>`_ for the full code