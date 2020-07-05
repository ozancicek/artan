Online Nonlinear Regression with Extended Kalman Filter
=======================================================

Exctended Kalman Filter (EKF) can be used for systems where measurement or state process updates are nonlinear
functions. In order to do nonlinear updates with EKF, the update function and its jacobian
must be specified.

To demonstrate a simple nonlinear example, the following generalized linear model with log link & gaussian noise is used.

     .. math::
        Y_t &= \exp(\beta X_t) + \epsilon : \epsilon \sim N(0, \sigma) \quad t=0,1,..T

The above model can be represented in state-space form by:

    .. math::

        V_t &= A_t V_{t-1} + u_t + q_t : q_t \sim N(0, Q) \\
        Z_t &= H_t(V_t) + r_t: r_t \sim N(0, R) \\

        A_t &= I \\
        u_t &= 0 \\
        Q &= 0 \\
        Z_t &= Y_t \\
        R &= \sigma \\
        H_t(V_t) &= \exp(V_t X_t) \\
        \frac{\partial H_t}{\partial V_t} &= X_t \exp(V_t X_t)


The process updates are linear whereas measurement updates are nonlinear.


Import EKF and start spark session.

    .. code-block:: scala

        import com.github.ozancicek.artan.ml.filter.ExtendedKalmanFilter
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions._
        import org.apache.spark.ml.linalg._

        val rowsPerSecond = 10
        val numStates = 10


Define the model parameters and udf's to generate training data.

For EKF, it is necessary define the nonlinear function and its jacobian if there is any. Only the measurement function
is nonlinear in this example, so it's enough to define the function mapping the state to measurement and
measurement jacobian.

In order to help these functions define evolving behaviour across measurements, they also accept `processModel` or `measurementModel`
as a second argument. So the signature of the function must be  ``(Vector, Matrix) => Vector`` for the nonlinear
function and ``(Vector, Matrix) => Matrix`` for its jacobian. The second argument to these functions can be
set with ``setMeasurementModelCol``or ``setProcessModelCol``. In this example, measurement model is used
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
          .setInitialStateCovariance(
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

        /**
        * -------------------------------------------
        * Batch: 2
        * -------------------------------------------
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        * |modelID|stateIndex|           stateMean|     stateCovariance|        residualMean|  residualCovariance|         mahalanobis|
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        * |      0|         5|[-0.0170639651961...|0.184650735418856...|[-0.0010775678634...| 21.24279669719657  |2.337969194146342E-4|
        * |      0|         6|[0.13372113418410...|0.097270109221418...|[2.3866966781327466]|21.892368858374287  |  0.5100947459174262|
        * |      1|         5|[0.21727975764867...|0.184289044729487...|[2.1590034862902434]| 20.72475537603141  | 0.47425141689857636|
        * |      1|         6|[0.16619831285685...|0.061682057710189...|[-1.0041419082389...|47.378255003177436  | 0.14588329445602757|
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        *
        * -------------------------------------------
        * Batch: 3
        * -------------------------------------------
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        * |modelID|stateIndex|           stateMean|     stateCovariance|        residualMean|  residualCovariance|         mahalanobis|
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        * |      0|         7|[0.21489917361592...|0.033224082430061...|[2.0552241094850023]| 41.05191755271204  | 0.32076905295206193|
        * |      0|         8|[0.20921262270095...|0.013189448768817...|[-0.2695123923053...| 45.00295378232299  |0.040175216810467415|
        * |      1|         7|[0.18172674610899...|0.031522374731488...|[0.4671830982405272]| 27.29893710175946  | 0.08941579732539723|
        * |      1|         8|[0.19249146732117...|0.016052060247902...|[0.4615553206598477]|28.440753092452763  | 0.08654723860064477|
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+--------------------+
        *
        * -------------------------------------------
        * Batch: 4
        * -------------------------------------------
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
        * |modelID|stateIndex|           stateMean|     stateCovariance|        residualMean|  residualCovariance|        mahalanobis|
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
        * |      0|         9|[0.18171784672603...|0.007654793457034...|[-1.9635172993212...| 28.22667169246637  |0.36957696607714374|
        * |      1|         9|[0.17499288278196...|0.008676615020153...|[-1.070230083612481]|27.589047780543666  |0.20375524590073577|
        * +-------+----------+--------------------+--------------------+--------------------+--------------------+-------------------+
        */
See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/ozancicek/artan/examples/streaming/EKFRateSourceGLMLog.scala>`_ for the full code