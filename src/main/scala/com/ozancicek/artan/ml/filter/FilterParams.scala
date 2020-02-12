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


/**
 * Param for initial state vector of a filter.
 */
private[artan] trait HasInitialState extends Params {

  def stateSize: Int

  final val initialState: Param[Vector] = new Param[Vector](
    this,
    "initialState",
    "Initial value of the state vector",
    (in: Vector) => in.size == stateSize)

  setDefault(initialState, new DenseVector(Array.fill(stateSize) {0.0}))

  final def getInitialState: Vector = $(initialState)

}


/**
 * Param for initial covariance of the state
 */
private[artan] trait HasInitialCovariance extends Params {

  def stateSize: Int

  final val initialCovariance: Param[Matrix] = new Param[Matrix](
    this,
    "initialCovariance",
    "Initial covariance matrix",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(initialCovariance, DenseMatrix.eye(stateSize))

  final def getInitialCovariance: Matrix = $(initialCovariance)

}


/**
 * Param for process model matrix
 */
private[artan] trait HasProcessModel extends Params {

  def stateSize: Int

  final val processModel: Param[Matrix] = new Param[Matrix](
    this,
    "processModel",
    "Process model matrix, transitions the state to the next state when applied",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(processModel, DenseMatrix.eye(stateSize))

  final def getProcessModel: Matrix = $(processModel)
}


/**
 * Param for fading factor
 */
private[artan] trait HasFadingFactor extends Params {

  final val fadingFactor: DoubleParam = new DoubleParam(
    this,
    "fadingFactor",
    "Factor controlling the weight of older measurements. With larger factor, more weights will" +
    "be given to recent measurements. Typically, should be really close to 1",
    ParamValidators.gtEq(1.0))

  setDefault(fadingFactor, 1.0)

  final def getFadingFactor: Double = $(fadingFactor)
}


/**
 * Param for measurement model matrix
 */
private[artan] trait HasMeasurementModel extends Params {

  def stateSize: Int
  def measurementSize: Int

  final val measurementModel: Param[Matrix] = new Param[Matrix](
    this,
    "measurementModel",
    "Measurement matrix, when multiplied with the state it should give the measurement vector",
    (in: Matrix) => (in.numRows == measurementSize) & (in.numCols == stateSize))

  setDefault(
    measurementModel,
    new DenseMatrix(
      measurementSize,
      stateSize,
      1.0 +: Array.fill(stateSize * measurementSize - 1) {0.0}))

  final def getMeasurementModel: Matrix = $(measurementModel)

}


/**
 * Param for process noise matrix
 */
private[artan] trait HasProcessNoise extends Params {

  def stateSize: Int

  final val processNoise: Param[Matrix] = new Param[Matrix](
    this,
    "processNoise",
    "Process noise matrix")

  setDefault(processNoise, DenseMatrix.eye(stateSize))

  final def getProcessNoise: Matrix = $(processNoise)
}


/**
 * Param for measurement noise matrix
 */
private[artan] trait HasMeasurementNoise extends Params {

  def measurementSize: Int

  final val measurementNoise: Param[Matrix] = new Param[Matrix](
    this,
    "measurementNoise",
    "Measurement noise matrix")

  setDefault(measurementNoise, DenseMatrix.eye(measurementSize))

  final def getMeasurementNoise: Matrix = $(measurementNoise)
}


/**
 * Param for process function, typically for nonlinear state transition processes
 */
private[artan] trait HasProcessFunction extends Params {

  final val processFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "processFunction",
    "Process function for nonlinear state transition. Input args to the function are state vector" +
    "and process model matrix. It should output the next state vector"
  )

  final def getProcessFunctionOpt: Option[(Vector, Matrix) => Vector] = get(processFunction)
}


/**
 * Param for process state jacobian for nonlinear state transition process
 */
private[artan] trait HasProcessStateJacobian extends Params {

  final val processStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processStateJacobian",
    "Process state jacobian function. Input args to the function are state vector" +
    "and process model matrix. It should output the jacobian matrix at input state"
  )

  final def getProcessStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processStateJacobian)
}


/**
 * Param for process noise jacobian for non-additive process noise
 */
private[artan] trait HasProcessNoiseJacobian extends Params {

  final val processNoiseJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processNoiseJacobian",
    "Process noise jacobian function. Input args to the function are the current state and process noise," +
    "it should output the process noise jacobian at input state. The output dimensions of the noise jacobian matrix" +
    "should be (n_state, n_noise), where n_state is the length of state vector and n_noise is the dimension of" +
    "the square processNoise matrix. Namely, let Q be an (n_noise, n_noise) matrix for process noise, and" +
    "let H_k(x_k) be the noise jacobian at state x_k. The result of H_k(x_k) * Q * H_k(x_k).T should have" +
    "(n_state, n_state) dimensions"
  )

  final def getProcessNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processNoiseJacobian)
}


/**
 * Param for measurement function, typically for nonlinear measurement equations.
 */
private[artan] trait HasMeasurementFunction extends Params {

  final val measurementFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "measurementFunction",
    "Measurement function for nonlinear measurement equations. Input args to the function are current" +
    "state and measurement model matrix. It should output the measurement corresponding to the state"
  )

  final def getMeasurementFunctionOpt: Option[(Vector, Matrix) => Vector] = get(measurementFunction)
}


/**
 * Param for measurement jacobian for nonlinear measurement equations
 */
private[artan] trait HasMeasurementStateJacobian extends Params {

  final val measurementStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementStateJacobian",
    "Measurement state jacobian function. Input args to the function are current state and measurement model" +
    "matrix. It should output the jacobian of measurement equation at input state"
  )

  final def getMeasurementStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementStateJacobian)
}


/**
 * Param for measurement noise jacobian for non-additive process noise
 */
private[artan] trait HasMeasurementNoiseJacobian extends Params {

  final val measurementNoiseJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementNoiseJacobian",
    "Measurement noise jacobian function. Input args to the function are the current state and" +
      "measurement noise, it should output the measurement noise jacobian at input state. " +
      "The output dimensions of the measurement noise jacobian matrix should be (n_measurement, n_noise)," +
      "where n_measurement is the length of measurement vector and n_noise is the dimension of" +
      "the square measurementNoise matrix. Namely, let Q be an (n_noise, n_noise) matrix for process noise, and" +
      "let H_k(x_k) be the noise jacobian at state x_k. The result of H_k(x_k) * Q * H_k(x_k).T should have" +
      "(n_measurement, n_measurement) dimensions"
  )

  final def getMeasurementNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementNoiseJacobian)
}

/**
 * Param for initial state column
 */
private[artan] trait HasInitialStateCol extends Params {

  final val initialStateCol: Param[String] = new Param[String](
    this,
    "initialStateCol",
    "Column name for initial state vector")

  final def getInitialStateCol: String = $(initialStateCol)
}


/**
 * Param for initial state covariance column
 */
private[artan] trait HasInitialCovarianceCol extends Params {

  final val initialCovarianceCol: Param[String] = new Param[String](
    this,
    "initialCovarianceCol",
    "Column name for initial covariance matrix")

  final def getInitialCovarianceCol: String = $(initialCovarianceCol)
}

/**
 * Param for measurement column
 */
private[artan] trait HasMeasurementCol extends Params {

  final val measurementCol: Param[String] = new Param[String](
    this,
    "measurementCol",
    "Column name for measurement vector. Missing measurements are allowed with nulls in the data")

  setDefault(measurementCol, "measurement")

  final def getMeasurementCol: String = $(measurementCol)
}


/**
 * Param for measurement model column
 */
private[artan] trait HasMeasurementModelCol extends Params {

  final val measurementModelCol: Param[String] = new Param[String](
    this,
    "measurementModelCol",
    "Column name for specifying measurement model from input DataFrame rather than" +
    "a constant measurement model for all filters")

  final def getMeasurementModelCol: String = $(measurementModelCol)
}


/**
 * Param for measurement noise column
 */
private[artan] trait HasMeasurementNoiseCol extends Params {

  final val measurementNoiseCol: Param[String] = new Param[String](
    this,
    "measurementNoiseCol",
    "Column name for specifying measurement noise from input DataFrame rather than" +
    "a constant measurement noise for all filters")

  final def getMeasurementNoiseCol: String = $(measurementNoiseCol)
}


/**
 * Param for process model column
 */
private[artan] trait HasProcessModelCol extends Params {

  final val processModelCol: Param[String] = new Param[String](
    this,
    "processModelCol",
    "Column name for specifying process model from input DataFrame rather than" +
    "a constant process model for all filters")

  final def getProcessModelCol: String = $(processModelCol)
}

/**
 * Param for process noise column
 */
private[artan] trait HasProcessNoiseCol extends Params {

  final val processNoiseCol: Param[String] = new Param[String](
    this,
    "processNoiseCol",
    "Column name for specifying process noise matrix from input DataFrame rather than" +
    "a constant process noise for all filters")

  final def getProcessNoiseCol: String = $(processNoiseCol)
}


/**
 * Param for control column
 */
private[artan] trait HasControlCol extends Params {

  final val controlCol: Param[String] = new Param[String](
    this,
    "controlCol",
    "Column name for specifying control vector")

  final def getControlCol: String = $(controlCol)
}


/**
 * Param for control function column
 */
private[artan] trait HasControlFunctionCol extends Params {

  final val controlFunctionCol: Param[String] = new Param[String](
    this,
    "controlFunctionCol",
    "Column name for specifying control matrix")

  final def getControlFunctionCol: String = $(controlFunctionCol)
}


/**
 * Param for enabling mahalanobis calculation
 */
private[artan] trait HasCalculateMahalanobis extends Params {

  final val calculateMahalanobis: BooleanParam = new BooleanParam(
    this,
    "calculateMahalanobis",
    "When true, mahalanobis distance of residual will be calculated & added to output DataFrame." +
    "Default is false.")

  setDefault(calculateMahalanobis, false)

  final def getCalculateMahalanobis: Boolean = $(calculateMahalanobis)
}


/**
 * Param for enabling loglikelihood calculation
 */
private[artan] trait HasCalculateLoglikelihood extends Params {

  final val calculateLoglikelihood: BooleanParam = new BooleanParam(
    this,
    "calculateLoglikelihood",
    "When true, loglikelihood of residual will be calculated & added to output DataFrame. Default is false")

  setDefault(calculateLoglikelihood, false)

  final def getCalculateLoglikelihood: Boolean = $(calculateLoglikelihood)
}