Online Nonlinear Regression with Extended Kalman Filter
=======================================================

EKF can be used for systems where measurement or state process updates are nonlinear
functions. In order to do nonlinear updates with EKF, the update function along and its jacobian
must be specified.

To demonstrate a simple nonlinear example, the following generalized linear model with log link & gaussian noise is used.

    .. image:: https://latex.codecogs.com/svg.latex?%5C%5C%20Y_t%3D%5Cexp%28%5Cbeta%20X_t%29%20&plus;%20%5Cepsilon%20%3A%20e%20%5Csim%20N%280%2C%20R%29

The above model can be represented in state-space form by:

    .. image:: https://latex.codecogs.com/svg.latex?%5C%5C%20V_t%20%3D%20A_t%20V_%7Bt-1%7D%20&plus;%20q_t%20%3A%20q_t%20%5Csim%20N%280%2C%20Q%29%20%5Cquad%20%28state%20%5C%20process%20%5C%20equation%29%5C%5C%20Z_t%20%3D%20H_t%28V_t%29%20&plus;%20r_t%20%3A%20r_t%20%5Csim%20N%280%2C%20R%29%20%5Cquad%20%28measurement%20%5C%20equation%29%5C%5C%20%5C%5C%20A_t%20%3D%20I%5C%5C%20q_t%20%3D%200%5C%5C%20H_t%28V_t%29%20%3D%20%5Cexp%28X_tV_t%29%5C%5C%20%5Cfrac%7B%5Cpartial%20H_t%7D%7B%5Cpartial%20V_t%7D%20%3D%20X_t%5Cexp%28X_tV_t%29%5C%5C

The process updates are linear whereas measurement updates are nonlinear.

Scala
-----

Import EKF and start spark session.

    .. code-block:: scala

        import com.github.ozancicek.artan.ml.filter.ExtendedKalmanFilter
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._
        import org.apache.spark.ml.linalg._

        val rowsPerSecond = 10
        val numStates = 10


Define the model parameters and udf's to generate training data.

For EKF, we need to define the nonlinear function and its jacobian if there is any. In this example, only the measurement function
is nonlinear, so it's enough to define the function mapping the state to measurement and measurement jacobian.

In order to help these functions define evolving behaviour across measurements, they also accept `processModel` or `measurementModel`
as a second argument. So the signature of the function must be  ``(Vector, Matrix) => Vector`` for the nonlinear
function and ``(Vector, Matrix) => Matrix`` for its jacobian. The second argument to these functions can be
set with ``setMeasurementModelCol```or ```setProcessModelCol``. In this example, measurement model is used
for defining the features matrix, and the nonlinear update is done with the defined function.

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

        // Measurement function and its jacobian
        val measurementFunc = (in: Vector, model: Matrix) => {
          val measurement = model.multiply(in)
          measurement.values(0) = scala.math.exp(measurement.values(0))
          measurement
        }

        val measurementJac = (in: Vector, model: Matrix) => {
          val dot = model.multiply(in)
          val res = scala.math.exp(dot(0))
          val jacs = Array(
            model(0, 0) * res,
            res
          )
          new DenseMatrix(1, 2, jacs)
        }

        val filter = new ExtendedKalmanFilter(stateSize, measurementSize)
          .setStateKeyCol("stateKey")
          .setInitialCovariance(
            new DenseMatrix(2, 2, Array(10.0, 0.0, 0.0, 10.0)))
          .setMeasurementCol("measurement")
          .setMeasurementModelCol("measurementModel")
          .setProcessModel(DenseMatrix.eye(2))
          .setProcessNoise(DenseMatrix.zeros(2, 2))
          .setMeasurementNoise(new DenseMatrix(1, 1, Array(10)))
          .setMeasurementFunction(measurementFunc)
          .setMeasurementStateJacobian(measurementJac)
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
          .queryName("EKFRateSourceGLMLog")
          .outputMode("append")
          .format("console")
          .start()

        query.awaitTermination()
        /*
        -------------------------------------------
        Batch: 32
        -------------------------------------------
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        |stateKey|stateIndex|               state|     stateCovariance|            residual|  residualCovariance|         mahalanobis|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        |       7|        38|[0.19992090504824...|4.91468217345624E...|[2.4948770866390078]|    22.23940688162  |  0.5290388359631079|
        |       3|        38|[0.19989311841702...|4.922383482303518...|[0.38183503107029...|22.260329711033293  | 0.08093008070411575|
        |       8|        38|[0.20009908402403...|4.929879485243636...|[0.6054629292293612]|22.265481361063664  | 0.12831325240765706|
        |       0|        38|[0.20009364771974...|4.926892465837449...|[-1.3858647755905...|22.258607277053002  | 0.29374593340097577|
        |       5|        38|[0.19649944366060...|5.451817669273846...| [40.40692021442874]|22.472870851169958  |   8.523666953468213|
        |       6|        38|[0.20009003997847...|4.938221750921417...|[1.3760025367041635]|22.274563576540046  |  0.2915510653366337|
        |       9|        38|[0.19998455876046...|4.911157950388761...|[0.16549119462433...| 22.24209342368559  |0.035090298345645275|
        |       1|        38|[0.19991617400097...|4.922928157769797...|[-0.9743583980571...|22.248084658699554  | 0.20657245861592055|
        |       4|        38|[0.19932844200826...|5.187338948824865...| [9.166036626837922]| 22.45859503419932  |  1.9341506419984322|
        |       2|        38|[0.20002416178557...|4.928373763795486...|[0.7331031952312514]|22.259913130053647  | 0.15538295621883577|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+

        -------------------------------------------
        Batch: 33
        -------------------------------------------
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        |stateKey|stateIndex|               state|     stateCovariance|            residual|  residualCovariance|         mahalanobis|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        |       7|        39|[0.19986140736208...|3.295259304552763...|[-2.205466193963275]|22.251583652305868  |  0.4675412595869415|
        |       3|        39|[0.19988586047713...|3.299207018008777...|[-0.2687453910489...| 22.25466453060011  |0.056967937836381155|
        |       8|        39|[0.20003374302277...|3.303298635599312...|[-2.417580181410358]|22.267212725376538  |   0.512327841866893|
        |       0|        39|[0.20007288286995...|3.302274105112857...|[-0.7685630367955...|  22.2562892371635  | 0.16291201501166258|
        |       5|        39|[0.19751438074470...|3.650511029811714...|   [35.912855197721]| 22.55325738661803  |   7.562150151089916|
        |       6|        39|[0.20000270958848...|3.307250218635134...|[-3.2276859061480...| 22.27912413586803  |  0.6838206464334063|
        |       9|        39|[0.19997516291702...|3.293548390788712...|[-0.3484231252477...|22.244164910530685  | 0.07387524239268677|
        |       1|        39|[0.19999054167496...|3.301238074687831...|[2.7542840207343033]|22.244206066320075  |  0.5839830845729057|
        |       4|        39|[0.19956490203243...|3.459105565657501...| [8.523849991371662]|22.457350992679096  |  1.7986908885931459|
        |       2|        39|[0.19996239798613...|3.304010769335991...|[-2.2864995734153...|22.261648864529647  |  0.4846100992211099|
        +--------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+

        */

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/ozancicek/artan/examples/streaming/EKFRateSourceGLMLog.scala>`_ for the full code