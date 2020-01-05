package com.ozancicek.artan.ml.filter

import com.ozancicek.artan.ml.state.{KalmanState, KalmanUpdate}
import com.ozancicek.artan.ml.state.{StateUpdateFunction, StatefulTransformer}
import com.ozancicek.artan.ml.stats.{MultivariateGaussian}
import com.ozancicek.artan.ml.linalg.{LinalgUtils}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Vectors, Matrices, Matrix}
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{lit, col, udf}
import org.apache.spark.sql.types._


trait HasStateMean extends Params {

  def stateSize: Int

  final val stateMean: Param[Vector] = new Param[Vector](
    this,
    "stateMean",
    " state mean",
    (in: Vector) => in.size == stateSize)

  setDefault(stateMean, new DenseVector(Array.fill(stateSize) {0.0}))

  final def getStateMean: Vector = $(stateMean)

}


trait HasStateCovariance extends Params {

  def stateSize: Int

  final val stateCov: Param[Matrix] = new Param[Matrix](
    this,
    "stateCov",
    "state covariance",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(stateCov, DenseMatrix.eye(stateSize))

  final def getStateCov: Matrix = $(stateCov)

}


trait HasProcessModel extends Params {

  def stateSize: Int

  final val processModel: Param[Matrix] = new Param[Matrix](
    this,
    "processModel",
    "process model",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(processModel, DenseMatrix.eye(stateSize))

  final def getProcessModel: Matrix = $(processModel)
}


trait HasFadingFactor extends Params {

  final val fadingFactor: Param[Double] = new DoubleParam(
    this,
    "fadingFactor",
    "Fading factor",
    ParamValidators.gtEq(1.0))

  setDefault(fadingFactor, 1.0)

  final def getFadingFactor: Double = $(fadingFactor)
}


trait HasMeasurementModel extends Params {

  def stateSize: Int
  def measurementSize: Int

  final val measurementModel: Param[Matrix] = new Param[Matrix](
    this,
    "measurementModel",
    "measurement model",
    (in: Matrix) => (in.numRows == measurementSize) & (in.numCols == stateSize))

  setDefault(
    measurementModel,
    new DenseMatrix(
      measurementSize,
      stateSize,
      1.0 +: Array.fill(stateSize * measurementSize - 1) {0.0}))

  final def getMeasurementModel: Matrix = $(measurementModel)

}


trait HasProcessNoise extends Params {

  def stateSize: Int

  final val processNoise: Param[Matrix] = new Param[Matrix](
    this,
    "processNoise",
    " process noise")

  setDefault(processNoise, DenseMatrix.eye(stateSize))

  final def getProcessNoise: Matrix = $(processNoise)
}


trait HasMeasurementNoise extends Params {

  def measurementSize: Int

  final val measurementNoise: Param[Matrix] = new Param[Matrix](
    this,
    "measurementNoise",
    " measurement noise")

  setDefault(measurementNoise, DenseMatrix.eye(measurementSize))

  final def getMeasurementNoise: Matrix = $(measurementNoise)
}


trait HasProcessFunction extends Params {

  final val processFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "processFunction",
    "Process Function"
  )

  final def getProcessFunctionOpt: Option[(Vector, Matrix) => Vector] = get(processFunction)
}


trait HasProcessStateJacobian extends Params {

  final val processStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processStateJacobian",
    "Process State Jacobian"
  )

  final def getProcessStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processStateJacobian)
}


trait HasProcessNoiseJacobian extends Params {

  final val processNoiseJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processNoiseJacobian",
    "Process Noise Jacobian"
  )

  final def getProcessNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processNoiseJacobian)
}


trait HasMeasurementFunction extends Params {

  final val measurementFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "measurementFunction",
    "Measurement Function"
  )

  final def getMeasurementFunctionOpt: Option[(Vector, Matrix) => Vector] = get(measurementFunction)
}


trait HasMeasurementStateJacobian extends Params {

  final val measurementStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementStateJacobian",
    "Measurement State Jacobian"
  )

  final def getMeasurementStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementStateJacobian)
}


trait HasMeasurementNoiseJacobian extends Params {

  final val measurementNoiseJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementNoiseJacobian",
    "Measurement Noise Jacobian"
  )

  final def getMeasurementNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementNoiseJacobian)
}


trait HasGroupKeyCol extends Params {

  final val groupKeyCol: Param[String] = new Param[String](
    this, "groupKeyCol", "group key column name")

  final def getGroupKeyCol: String = $(groupKeyCol)
}


trait HasMeasurementCol extends Params {

  final val measurementCol: Param[String] = new Param[String](
    this, "measurementCol", "measurement columnn")

  final def getMeasurementCol: String = $(measurementCol)
}


trait HasMeasurementModelCol extends Params {

  final val measurementModelCol: Param[String] = new Param[String](
    this, "measurementModelCol", "measurement model columnn")

  final def getMeasurementModelCol: String = $(measurementModelCol)
}


trait HasMeasurementNoiseCol extends Params {

  final val measurementNoiseCol: Param[String] = new Param[String](
    this, "measurementNoiseCol", "measurement model columnn")

  final def getMeasurementNoiseCol: String = $(measurementNoiseCol)
}


trait HasProcessModelCol extends Params {

  final val processModelCol: Param[String] = new Param[String](
    this, "processModelCol", "process model columnn")

  final def getProcessModelCol: String = $(processModelCol)
}


trait HasProcessNoiseCol extends Params {

  final val processNoiseCol: Param[String] = new Param[String](
    this, "processNoiseCol", "process noise columnn")

  final def getProcessNoiseCol: String = $(processNoiseCol)
}


trait HasControlCol extends Params {

  final val controlCol: Param[String] = new Param[String](
    this, "controlCol", "control columnn")

  final def getControlCol: String = $(controlCol)
}


trait HasControlFunctionCol extends Params {

  final val controlFunctionCol: Param[String] = new Param[String](
    this, "controlFunctionCol", "control function columnn")

  final def getControlFunctionCol: String = $(controlFunctionCol)
}


trait HasCalculateMahalanobis extends Params {

  final val calculateMahalanobis: Param[Boolean] = new Param[Boolean](
    this, "calculateMahalanobis", "calculate mahalanobis")

  setDefault(calculateMahalanobis, false)

  final def getCalculateMahalanobis: Boolean = $(calculateMahalanobis)
}


trait HasCalculateLoglikelihood extends Params {

  final val calculateLoglikelihood: Param[Boolean] = new Param[Boolean](
    this, "calculateLoglikelihood", "calculate loglikelihood")

  setDefault(calculateLoglikelihood, false)

  final def getCalculateLoglikelihood: Boolean = $(calculateLoglikelihood)
}


trait KalmanUpdateParams extends HasGroupKeyCol with HasMeasurementCol
  with HasMeasurementModelCol with HasMeasurementNoiseCol
  with HasProcessModelCol with HasProcessNoiseCol with HasControlCol
  with HasControlFunctionCol with HasProcessModel with HasMeasurementModel
  with HasProcessNoise with HasMeasurementNoise
  with HasCalculateMahalanobis with HasCalculateLoglikelihood {

  protected def getGroupKeyExpr = col($(groupKeyCol)).cast(StringType)

  protected def getMeasurementExpr = col($(measurementCol)).cast(SQLDataTypes.VectorType)

  protected def getMeasurementModelExpr = {
   if (isSet(measurementModelCol)) {
     col($(measurementModelCol))
   } else {
     val default = $(measurementModel)
     val col = udf(()=>default)
     col()
   }
  }

  protected def getMeasurementNoiseExpr = {
   if (isSet(measurementNoiseCol)) {
     col($(measurementNoiseCol))
   } else {
     val default = $(measurementNoise)
     val col = udf(()=>default)
     col()
   }
  }

  protected def getProcessModelExpr = {
   if (isSet(processModelCol)) {
     col($(processModelCol))
   } else {
     val default = $(processModel)
     val col = udf(()=>default)
     col()
   }
  }

  protected def getProcessNoiseExpr = {
   if (isSet(processNoiseCol)) {
     col($(processNoiseCol))
   } else {
     val default = $(processNoise)
     val col = udf(()=>default)
     col()
   }
  }

  protected def getControlExpr = {
   if (isSet(controlCol)) {
     col($(controlCol))
   } else {
     lit(null).cast(SQLDataTypes.VectorType)
   }
  }

  protected def getControlFunctionExpr = {
    if (isSet(controlFunctionCol)) {
      col($(controlFunctionCol))
    } else {
      lit(null).cast(SQLDataTypes.MatrixType)
    }
  }

  protected def validateSchema(schema: StructType): Unit = {
    require(isSet(groupKeyCol), "Group key column must be set")
    require(schema($(groupKeyCol)).dataType == StringType, "Group key column must be StringType")
    require(isSet(measurementCol), "Measurement column must be set")

    if (isSet(measurementModelCol)) {
      require(
        schema($(measurementModelCol)).dataType == SQLDataTypes.MatrixType,
        "Measurement model column must be MatrixType")
    }

    val vectorCols = Seq(measurementCol, controlCol)
    val matrixCols = Seq(
      measurementModelCol, measurementNoiseCol, processModelCol,
      processNoiseCol, controlFunctionCol)

    vectorCols.foreach(col=>validateColParamType(schema, col, SQLDataTypes.VectorType))
    matrixCols.foreach(col=>validateColParamType(schema, col, SQLDataTypes.MatrixType))
  }

  private def validateColParamType(schema: StructType, col: Param[String], t: DataType): Unit = {
    if (isSet(col)) {
      val colname = $(col)
      require(schema(colname).dataType == t,
              s"$colname must be of $t")
    }
  }

}


private[ml] trait KalmanStateCompute extends Serializable {

  private[ml] def update(
    state: KalmanState,
    process: KalmanUpdate): KalmanState

  private[ml] def predict(
    state: KalmanState,
    process: KalmanUpdate): KalmanState

  def logpdf(residual: DenseVector, uncertainity: DenseMatrix): Double = {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    MultivariateGaussian.logpdf(residual, zeroMean, uncertainity)
  }
}


private[ml] trait KalmanStateUpdateFunction[+Compute <: KalmanStateCompute]
  extends StateUpdateFunction[String, KalmanUpdate, KalmanState] {

  val kalmanCompute: Compute
  def stateMean: Vector
  def stateCov: Matrix

  def updateGroupState(
    key: String,
    row: KalmanUpdate,
    state: Option[KalmanState]): Option[KalmanState] = {
    
    val currentState = state
      .getOrElse(KalmanState(
         key,
         0L,
         stateMean.toDense,
         stateCov.toDense,
         new DenseVector(Array(0.0)),
         DenseMatrix.zeros(1, 1)))

    val nextState = row.measurement match {
      case Some(m) => kalmanCompute.update(currentState, row)
      case None => kalmanCompute.predict(currentState, row)
    }
    Some(nextState)
  }
}


private[ml] abstract class KalmanTransformer[
   Compute <: KalmanStateCompute,
   StateUpdate <: KalmanStateUpdateFunction[Compute]] 
   extends StatefulTransformer[String, KalmanUpdate, KalmanState]
   with KalmanUpdateParams {

  implicit val kalmanUpdateEncoder = Encoders.product[KalmanUpdate]
  implicit val groupKeyEncoder = Encoders.STRING
  implicit val kalmanStateEncoder = Encoders.product[KalmanState]

  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    kalmanUpdateEncoder.schema
  }

  def loglikelihoodUDF = udf((residual: Vector, covariance: Matrix) => {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    MultivariateGaussian.logpdf(residual.toDense, zeroMean, covariance.toDense)
  })

  def mahalanobisUDF = udf((residual: Vector, covariance: Matrix) => {
    val zeroMean = new DenseVector(Array.fill(residual.size) {0.0})
    LinalgUtils.mahalanobis(residual.toDense, zeroMean, covariance.toDense)
  }) 

  def withExtraColumns(dataset: Dataset[KalmanState]): DataFrame = {
    val df = dataset.toDF
    val withLoglikelihood = if (getCalculateLoglikelihood) {
      df.withColumn("loglikelihood", loglikelihoodUDF(col("residual"), col("residualCovariance")))
    } else {df}

    val withMahalanobis = if (getCalculateMahalanobis) {
      withLoglikelihood.withColumn("mahalanobis", mahalanobisUDF(col("residual"), col("residualCovariance")))
    } else {withLoglikelihood}

    withMahalanobis
  }

  def filter(dataset: Dataset[_]): Dataset[KalmanState] = {
    transformSchema(dataset.schema)
    val kalmanUpdateDS = toKalmanUpdate(dataset) 
    transformWithState(kalmanUpdateDS)
  }

  def toKalmanUpdate(dataset: Dataset[_]): Dataset[KalmanUpdate] = {
    dataset
      .withColumn("groupKey", getGroupKeyExpr)
      .withColumn("measurement", getMeasurementExpr)
      .withColumn("measurementModel", getMeasurementModelExpr)
      .withColumn("measurementNoise", getMeasurementNoiseExpr)
      .withColumn("processModel", getProcessModelExpr)
      .withColumn("processNoise", getProcessNoiseExpr)
      .withColumn("control", getControlExpr)
      .withColumn("controlFunction", getControlFunctionExpr)
      .select("groupKey", "measurement", "measurementModel",
              "measurementNoise", "processModel", "processNoise",
              "control", "controlFunction")
      .as(kalmanUpdateEncoder)
  }

  def keyFunc = (in: KalmanUpdate) => in.groupKey
  def stateUpdateFunc: StateUpdate

}
