/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ozancicek.artan.ml.filter

import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Matrix}
import org.apache.spark.ml.param._


private[filter] trait HasStateMean extends Params {

  def stateSize: Int

  final val stateMean: Param[Vector] = new Param[Vector](
    this,
    "stateMean",
    " state mean",
    (in: Vector) => in.size == stateSize)

  setDefault(stateMean, new DenseVector(Array.fill(stateSize) {0.0}))

  final def getStateMean: Vector = $(stateMean)

}


private[filter] trait HasStateCovariance extends Params {

  def stateSize: Int

  final val stateCov: Param[Matrix] = new Param[Matrix](
    this,
    "stateCov",
    "state covariance",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(stateCov, DenseMatrix.eye(stateSize))

  final def getStateCov: Matrix = $(stateCov)

}


private[filter] trait HasProcessModel extends Params {

  def stateSize: Int

  final val processModel: Param[Matrix] = new Param[Matrix](
    this,
    "processModel",
    "process model",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(processModel, DenseMatrix.eye(stateSize))

  final def getProcessModel: Matrix = $(processModel)
}


private[filter] trait HasFadingFactor extends Params {

  final val fadingFactor: DoubleParam = new DoubleParam(
    this,
    "fadingFactor",
    "Fading factor",
    ParamValidators.gtEq(1.0))

  setDefault(fadingFactor, 1.0)

  final def getFadingFactor: Double = $(fadingFactor)
}


private[filter] trait HasMeasurementModel extends Params {

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


private[filter] trait HasProcessNoise extends Params {

  def stateSize: Int

  final val processNoise: Param[Matrix] = new Param[Matrix](
    this,
    "processNoise",
    " process noise")

  setDefault(processNoise, DenseMatrix.eye(stateSize))

  final def getProcessNoise: Matrix = $(processNoise)
}


private[filter] trait HasMeasurementNoise extends Params {

  def measurementSize: Int

  final val measurementNoise: Param[Matrix] = new Param[Matrix](
    this,
    "measurementNoise",
    " measurement noise")

  setDefault(measurementNoise, DenseMatrix.eye(measurementSize))

  final def getMeasurementNoise: Matrix = $(measurementNoise)
}


private[filter] trait HasProcessFunction extends Params {

  final val processFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "processFunction",
    "Process Function"
  )

  final def getProcessFunctionOpt: Option[(Vector, Matrix) => Vector] = get(processFunction)
}


private[filter] trait HasProcessStateJacobian extends Params {

  final val processStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processStateJacobian",
    "Process State Jacobian"
  )

  final def getProcessStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processStateJacobian)
}


private[filter] trait HasProcessNoiseJacobian extends Params {

  final val processNoiseJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processNoiseJacobian",
    "Process Noise Jacobian"
  )

  final def getProcessNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processNoiseJacobian)
}


private[filter] trait HasMeasurementFunction extends Params {

  final val measurementFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "measurementFunction",
    "Measurement Function"
  )

  final def getMeasurementFunctionOpt: Option[(Vector, Matrix) => Vector] = get(measurementFunction)
}


private[filter] trait HasMeasurementStateJacobian extends Params {

  final val measurementStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementStateJacobian",
    "Measurement State Jacobian"
  )

  final def getMeasurementStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementStateJacobian)
}


private[filter] trait HasMeasurementNoiseJacobian extends Params {

  final val measurementNoiseJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementNoiseJacobian",
    "Measurement Noise Jacobian"
  )

  final def getMeasurementNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementNoiseJacobian)
}


private[filter] trait HasGroupKeyCol extends Params {

  final val groupKeyCol: Param[String] = new Param[String](
    this, "groupKeyCol", "group key column name")

  final def getGroupKeyCol: String = $(groupKeyCol)
}


private[filter] trait HasMeasurementCol extends Params {

  final val measurementCol: Param[String] = new Param[String](
    this, "measurementCol", "measurement column")

  setDefault(measurementCol, "measurement")

  final def getMeasurementCol: String = $(measurementCol)
}


private[filter] trait HasMeasurementModelCol extends Params {

  final val measurementModelCol: Param[String] = new Param[String](
    this, "measurementModelCol", "measurement model columnn")

  final def getMeasurementModelCol: String = $(measurementModelCol)
}


private[filter] trait HasMeasurementNoiseCol extends Params {

  final val measurementNoiseCol: Param[String] = new Param[String](
    this, "measurementNoiseCol", "measurement model columnn")

  final def getMeasurementNoiseCol: String = $(measurementNoiseCol)
}


private[filter] trait HasProcessModelCol extends Params {

  final val processModelCol: Param[String] = new Param[String](
    this, "processModelCol", "process model columnn")

  final def getProcessModelCol: String = $(processModelCol)
}


private[filter] trait HasProcessNoiseCol extends Params {

  final val processNoiseCol: Param[String] = new Param[String](
    this, "processNoiseCol", "process noise columnn")

  final def getProcessNoiseCol: String = $(processNoiseCol)
}


private[filter] trait HasControlCol extends Params {

  final val controlCol: Param[String] = new Param[String](
    this, "controlCol", "control columnn")

  final def getControlCol: String = $(controlCol)
}


private[filter] trait HasControlFunctionCol extends Params {

  final val controlFunctionCol: Param[String] = new Param[String](
    this, "controlFunctionCol", "control function columnn")

  final def getControlFunctionCol: String = $(controlFunctionCol)
}


private[filter] trait HasCalculateMahalanobis extends Params {

  final val calculateMahalanobis: BooleanParam = new BooleanParam(
    this, "calculateMahalanobis", "calculate mahalanobis")

  setDefault(calculateMahalanobis, false)

  final def getCalculateMahalanobis: Boolean = $(calculateMahalanobis)
}


private[filter] trait HasCalculateLoglikelihood extends Params {

  final val calculateLoglikelihood: BooleanParam = new BooleanParam(
    this, "calculateLoglikelihood", "calculate loglikelihood")

  setDefault(calculateLoglikelihood, false)

  final def getCalculateLoglikelihood: Boolean = $(calculateLoglikelihood)
}