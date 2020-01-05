package com.ozancicek.artan.ml.filter
import breeze.stats.distributions.{RandBasis, Gaussian}
import com.ozancicek.artan.ml.testutils.SparkSessionTestWrapper
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.{LAPACK}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.scalatest.{FunSpec, Matchers}


class RecursiveLeastSquaresFilterSpec
  extends FunSpec
  with Matchers
  with SparkSessionTestWrapper {

  import spark.implicits._
  implicit val basis: RandBasis = RandBasis.withSeed(0)

  describe("Batch RLS tests") {
    it("should be equivalent to ols") {
      // Ols problem
      // z = a*x + b*y + c + N(0, R)
      val n = 100
      val dist = breeze.stats.distributions.Gaussian(0, 1)

      val a = 1.5
      val b = -2.7
      val c = 5.0
      val xs = (0 until n).map(_.toDouble).toArray
      val ys = (0 until n).map(i=> scala.math.sqrt(i.toDouble)).toArray
      val zs = xs.zip(ys).map {
        case(x,y)=> (x, y, a*x + b*y + c + dist.draw())
      }
      val df = zs.map {
        case (x, y, z) => ("1", z, new DenseVector(Array(x, y, 1)))
      }.toSeq.toDF("modelId", "label", "features")

      val filter = new RecursiveLeastSquaresFilter(3)
        .setGroupKeyCol("modelId")
        .setInverseCovarianceDiag(10000)

      val modelState = filter.transform(df)

      val lastState = modelState.collect
        .filter(row=>row.getAs[Long]("index") == n)(0)
        .getAs[DenseVector]("mean")

      // find least squares solution with dgels
      val features = new DenseMatrix(n, 3, xs ++ ys ++ Array.fill(n) {1.0})
      val target = new DenseVector(zs.map {case (x, y, z) => z}.toArray)
      val coeffs = LAPACK.dgels(features, target)
      // Error is mean absolute difference of kalman and least squares solutions
      val mae = (0 until coeffs.size).foldLeft(0.0) {
        case(s, i) => s + scala.math.abs(lastState(i) - coeffs(i))
      } / coeffs.size
      // Error should be smaller than a certain threshold. The threshold is
      // tuned to some arbitrary small value depending on noise, cov and true coefficients.
      val threshold = 1E-4

      assert(mae < threshold)
    }
  }
}
