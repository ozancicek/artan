package com.ozancicek.artan.ml.filter
import breeze.stats.distributions.{RandBasis, Gaussian}
import com.ozancicek.artan.ml.testutils.SparkSessionTestWrapper
import org.apache.spark.ml.BLAS
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.{LAPACK}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.scalatest.{FunSpec, Matchers}

class ExtendedKalmanFilterSpec
  extends FunSpec
  with Matchers
  with SparkSessionTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)

  describe("Batch extended kalman filter tests") {
    it("should estimate generalized linear regression model parameters") {
      // generalized linear regression with log link
      // z = exp(a*x + b*y + c) + N(0, 1)
      val a = 0.5
      val b = -0.7
      val c = 2.0
      val dist = breeze.stats.distributions.Gaussian(0, 1)
      val n = 40
      val xs = (0 until n).map(i=>i.toDouble).toArray
      val ys = (0 until n).map(i=> scala.math.sqrt(i.toDouble)).toArray
      val zs = xs.zip(ys).map {
        case(x,y)=> (x, y, scala.math.exp(a*x + b*y + c) + dist.draw())
      }
      val df = zs.map {
        case (x, y, z) => ("1", new DenseVector(Array(z)), new DenseMatrix(1, 3, Array(x, y, 1)))
      }.toSeq.toDF("modelId", "measurement", "measurementModel")

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
          model(0, 1) * res,
          res
        )
        new DenseMatrix(1, 3, jacs.toArray)
      }

      val filter = new ExtendedKalmanFilter(3, 1)
        .setGroupKeyCol("modelId")
        .setStateCovariance(
          new DenseMatrix(3, 3, Array(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)))
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModel(DenseMatrix.eye(3))
        .setProcessNoise(DenseMatrix.zeros(3, 3))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(10)))
        .setMeasurementFunction(measurementFunc)
        .setMeasurementStateJacobian(measurementJac)

      val modelState = filter.transform(df)

      val lastState = modelState.collect
        .filter(row=>row.getAs[Long]("index") == n)(0)
        .getAs[DenseVector]("mean")

      val coeffs = new DenseVector(Array(a, b, c))
      val mae = (0 until coeffs.size).foldLeft(0.0) {
        case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
      } / coeffs.size
      // Error should be smaller than a certain threshold. The threshold is
      // tuned to some arbitrary small value depending on noise, cov and true coefficients.
      val threshold = 1E-4
      assert(mae < threshold)
    }

    it("should filter ma model") {
      // ma(1) model
      // z(t) = b*eps(t-1) + eps(t)
      // eps(t) ~ N(0, R)

      val b = 0.8
      val dist = breeze.stats.distributions.Gaussian(0, 1)
      val n = 40
      var epsPrior = dist.draw()

      val zs = (0 until n).map { i=>
        val epsCurrent = dist.draw()
        val z = b*epsPrior + epsCurrent
        epsPrior = epsCurrent
        z
      }

      val df = zs.map { z=>
        val processModel = new DenseMatrix(
          2, 2, Array(0, 0, 1, 0))
        val measurement = new DenseVector(Array(z))
        val measurementModel = new DenseMatrix(1, 2, Array(1, 0))
        ("1", measurement, measurementModel, processModel)
      }.toDF("modelID", "measurement", "measurementModel", "processModel")

      val processNoiseJac = (in: Vector, model: Matrix) => {
        new DenseMatrix(2, 1, Array(1.0, 0.9))
      }

      val filter = new ExtendedKalmanFilter(2, 1)
        .setGroupKeyCol("modelID")
        .setMeasurementCol("measurement")
        .setMeasurementModelCol("measurementModel")
        .setProcessModelCol("processModel")
        .setProcessNoise(new DenseMatrix(1, 1, Array(1.0)))
        .setMeasurementNoise(new DenseMatrix(1, 1, Array(0.0001)))
        .setProcessNoiseJacobian(processNoiseJac)
        .setCalculateMahalanobis
        .setCalculateLoglikelihood

      val modelState = filter.transform(df)

      val covExtract = udf((in: Matrix) => in(0, 0))

      val stats = modelState
        .withColumn("residualCovariance", covExtract($"residualCovariance"))
        .groupBy($"groupKey")
        .agg(
          avg($"mahalanobis").alias("mahalanobis"),
          avg($"loglikelihood").alias("loglikelihood"),
          avg($"residualCovariance").alias("residualCovariance"),
          Summarizer.mean($"mean").alias("avg"))
        .head

      assert(scala.math.abs(stats.getAs[DenseVector]("avg")(0) - zs.reduce(_ + _)/zs.size) < 1.0)
      assert(stats.getAs[Double]("mahalanobis") < 2.0)
      assert(scala.math.abs(stats.getAs[Double]("residualCovariance") - 1.0) < 0.1)
    }

  }
}
