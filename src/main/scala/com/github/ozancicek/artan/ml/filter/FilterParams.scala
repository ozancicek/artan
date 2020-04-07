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

package com.github.ozancicek.artan.ml.filter

import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Matrix}
import org.apache.spark.ml.param._


/**
 * Param for initial state vector of a filter.
 */
private[artan] trait HasInitialState extends Params {

  def stateSize: Int

  /**
   * Param for initial value of the state vector
   *
   * @group param
   */
  final val initialState: Param[Vector] = new Param[Vector](
    this,
    "initialState",
    "Initial value of the state vector",
    (in: Vector) => in.size == stateSize)

  setDefault(initialState, new DenseVector(Array.fill(stateSize) {0.0}))

  /**
   * Getter for the initial value of the state vector
   *
   * @group getParam
   */
  final def getInitialState: Vector = $(initialState)

}


/**
 * Param for initial covariance of the state
 */
private[artan] trait HasInitialCovariance extends Params {

  def stateSize: Int

  /**
   * Param for the initial covariance matrix
   *
   * @group param
   */
  final val initialCovariance: Param[Matrix] = new Param[Matrix](
    this,
    "initialCovariance",
    "Initial covariance matrix",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(initialCovariance, DenseMatrix.eye(stateSize))

  /**
   * Getter for the initial covariance matrix param
   *
   * @group getParam
   */
  final def getInitialCovariance: Matrix = $(initialCovariance)

}


/**
 * Param for process model matrix
 */
private[artan] trait HasProcessModel extends Params {

  def stateSize: Int

  /**
   * Param for the process model matrix, transitions the state to the next state with dot product.
   *
   * @group param
   */
  final val processModel: Param[Matrix] = new Param[Matrix](
    this,
    "processModel",
    "Process model matrix, transitions the state to the next state with dot produc",
    (in: Matrix) => (in.numRows == stateSize) & (in.numCols == stateSize))

  setDefault(processModel, DenseMatrix.eye(stateSize))

  /**
   * Getter for the process model matrix param
   *
   * @group getParam
   */
  final def getProcessModel: Matrix = $(processModel)
}


/**
 * Param for fading factor
 */
private[artan] trait HasFadingFactor extends Params {

  /**
   * Param for controlling the weight of older measurements, should be >= 1.0 . A factor of 1.0 will result
   * in equal weight of all measurements. With larger factor, the weight of recent measurements will increase.
   * Typically set around 1.01 ~ 1.5
   *
   * Default is 1.0
   *
   * @group param
   */
  final val fadingFactor: DoubleParam = new DoubleParam(
    this,
    "fadingFactor",
    "Param for controlling the weight of older measurements, should be >= 1.0 . A factor of 1.0 will result" +
    "in equal weight of all measurements. With larger factor, the weight of recent measurements will increase." +
    "Typically set around 1.01 ~ 1.5",
    ParamValidators.gtEq(1.0))

  setDefault(fadingFactor, 1.0)

  /**
   * Getter for fading factor param
   *
   * @group getParam
   */
  final def getFadingFactor: Double = $(fadingFactor)
}


/**
 * Param for measurement model matrix
 */
private[artan] trait HasMeasurementModel extends Params {

  def stateSize: Int
  def measurementSize: Int

  /**
   * Param for measurement model matrix. Its dot-product with state should produce the measurement vector.
   *
   * @group param
   */
  final val measurementModel: Param[Matrix] = new Param[Matrix](
    this,
    "measurementModel",
    "Measurement matrix. Its dot-product with state should produce the measurement vector.",
    (in: Matrix) => (in.numRows == measurementSize) & (in.numCols == stateSize))

  setDefault(
    measurementModel,
    new DenseMatrix(
      measurementSize,
      stateSize,
      1.0 +: Array.fill(stateSize * measurementSize - 1) {0.0}))

  /**
   * Getter for measurement model matrix param
   * @return
   */
  final def getMeasurementModel: Matrix = $(measurementModel)

}


/**
 * Param for process noise matrix
 */
private[artan] trait HasProcessNoise extends Params {

  def stateSize: Int

  /**
   * Param for process noise covariance matrix, should be a square matrix with dimensions stateSize x stateSize.
   *
   * Default is identity matrix.
   *
   * @group param
   */
  final val processNoise: Param[Matrix] = new Param[Matrix](
    this,
    "processNoise",
    "Process noise covariance matrix, should be a square matrix with dimensions stateSize x stateSize")

  setDefault(processNoise, DenseMatrix.eye(stateSize))

  /**
   * Getter for process noise covariance matrix param
   *
   * @group getParam
   */
  final def getProcessNoise: Matrix = $(processNoise)
}


/**
 * Param for measurement noise covariance matrix
 */
private[artan] trait HasMeasurementNoise extends Params {

  def measurementSize: Int

  /**
   * Param for measurement noise covariance matrix, should be a square matrix with dimensions
   * measurementSize x measurementSize.
   *
   * @group param
   */
  final val measurementNoise: Param[Matrix] = new Param[Matrix](
    this,
    "measurementNoise",
    "Measurement noise matrix")

  setDefault(measurementNoise, DenseMatrix.eye(measurementSize))

  /**
   * Getter for measurement noise matrix param.
   *
   * @group getParam
   */
  final def getMeasurementNoise: Matrix = $(measurementNoise)
}


/**
 * Param for process function, typically for nonlinear state transition processes
 */
private[artan] trait HasProcessFunction extends Params {

  /**
   * Param for process function for nonlinear state transition. Input args to the function are state vector
   * and process model matrix. It should output the next state vector.
   *
   * @group param
   */
  final val processFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "processFunction",
    "Process function for nonlinear state transition. Input args to the function are state vector" +
    "and process model matrix. It should output the next state vector"
  )

  /**
   * Getter for process function param.
   *
   * @group getParam
   */
  final def getProcessFunctionOpt: Option[(Vector, Matrix) => Vector] = get(processFunction)
}


/**
 * Param for process state jacobian for nonlinear state transition process
 */
private[artan] trait HasProcessStateJacobian extends Params {

  /**
   * Param for process state jacobian function. Input args to the function are state vector
   * and process model matrix. It should output the jacobian matrix at input state.
   *
   * @group param
   */
  final val processStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "processStateJacobian",
    "Process state jacobian function. Input args to the function are state vector" +
    "and process model matrix. It should output the jacobian matrix at input state"
  )

  /**
   * Getter for process state jacobian function
   *
   * @group getParam
   */
  final def getProcessStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processStateJacobian)
}


/**
 * Param for process noise jacobian for non-additive process noise
 */
private[artan] trait HasProcessNoiseJacobian extends Params {

  /**
   * Param for process noise jacobian for non-additive process noise.
   *
   * Input args to the function are the current state and process noise,
   * it should output the process noise jacobian at input state. The output dimensions of the noise jacobian matrix
   * should be (n_state, n_noise), where n_state is the length of state vector and n_noise is the dimension of
   * the square processNoise matrix. Namely, let Q be an (n_noise, n_noise) matrix for process noise, and
   * let H_k(x_k) be the noise jacobian at state x_k. The result of H_k(x_k) * Q * H_k(x_k).T should have
   * (n_state, n_state) dimensions
   *
   * @group param
   */
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

  /**
   * Getter for process noise jacobian function.
   * @group getParam
   */
  final def getProcessNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(processNoiseJacobian)
}


/**
 * Param for measurement function, typically for nonlinear measurement equations.
 */
private[artan] trait HasMeasurementFunction extends Params {

  /**
   * Param for measurement function in nonlinear measurement equations. Input args to the function are current
   * state and measurement model matrix. It should output the measurement corresponding to the state
   * @group param
   */
  final val measurementFunction: Param[(Vector, Matrix) => Vector] = new Param[(Vector, Matrix) => Vector](
    this,
    "measurementFunction",
    "Measurement function for nonlinear measurement equations. Input args to the function are current" +
    "state and measurement model matrix. It should output the measurement corresponding to the state"
  )

  /**
   * Getter for measurement function param
   * @group getParam
   */
  final def getMeasurementFunctionOpt: Option[(Vector, Matrix) => Vector] = get(measurementFunction)
}


/**
 * Param for measurement jacobian for nonlinear measurement equations
 */
private[artan] trait HasMeasurementStateJacobian extends Params {

  /**
   * Param for measurement state jacobian function. Input args to the function are current state and measurement model"
   * matrix. It should output the jacobian of measurement equation at input state.
   *
   * @group param
   */
  final val measurementStateJacobian: Param[(Vector, Matrix) => Matrix] = new Param[(Vector, Matrix) => Matrix](
    this,
    "measurementStateJacobian",
    "Measurement state jacobian function. Input args to the function are current state and measurement model" +
    "matrix. It should output the jacobian of measurement equation at input state"
  )

  /**
   * Getter for measurement state jacobian function.
   * @group getParam
   */
  final def getMeasurementStateJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementStateJacobian)
}


/**
 * Param for measurement noise jacobian for non-additive measurement noise
 */
private[artan] trait HasMeasurementNoiseJacobian extends Params {

  /**
   * Param for measurement noise jacobian function for non-additive measurement noise.
   *
   * Input args to the function are the current state and
   * measurement noise, it should output the measurement noise jacobian at input state.
   * The output dimensions of the measurement noise jacobian matrix should be (n_measurement, n_noise),
   * where n_measurement is the length of measurement vector and n_noise is the dimension of
   * the square measurementNoise matrix. Namely, let Q be an (n_noise, n_noise) matrix for process noise, and
   * let H_k(x_k) be the noise jacobian at state x_k. The result of H_k(x_k) * Q * H_k(x_k).T should have
   * (n_measurement, n_measurement) dimensions
   *
   * @group param
   */
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

  /**
   * Getter for measurement noise jacobian function.
   * @group getParam
   */
  final def getMeasurementNoiseJacobianOpt: Option[(Vector, Matrix) => Matrix] = get(measurementNoiseJacobian)
}

/**
 * Param for initial state column
 */
private[artan] trait HasInitialStateCol extends Params {

  /**
   * Param for initial state vector as a dataframe column. Overrides [[initialState]] param. Can
   * be used for initializing separate state vector for each state.
   *
   * @group param
   */
  final val initialStateCol: Param[String] = new Param[String](
    this,
    "initialStateCol",
    "Column name for initial state vector")

  /**
   * Getter for initial state vector column
   * @group getParam
   */
  final def getInitialStateCol: String = $(initialStateCol)
}


/**
 * Param for initial state covariance column
 */
private[artan] trait HasInitialCovarianceCol extends Params {

  /**
   * Param for initial state covariance matrix as a dataframe column. Overrides [[initialCovariance]] param. Can
   * be used for initializing separate state covariance matrix for each state.
   *
   * @group param
   */
  final val initialCovarianceCol: Param[String] = new Param[String](
    this,
    "initialCovarianceCol",
    "Column name for initial covariance matrix")

  /**
   * Getter for initial state covariance matrix column
   *
   * @group getParam
   */
  final def getInitialCovarianceCol: String = $(initialCovarianceCol)
}

/**
 * Param for measurement column
 */
private[artan] trait HasMeasurementCol extends Params {

  /**
   * Param for measurement column containing measurement vector.
   * @group param
   */
  final val measurementCol: Param[String] = new Param[String](
    this,
    "measurementCol",
    "Column name for measurement vector. Missing measurements are allowed with nulls in the data")

  setDefault(measurementCol, "measurement")

  /**
   * Getter for measurement vector column.
   * @group getParam
   */
  final def getMeasurementCol: String = $(measurementCol)
}


/**
 * Param for measurement model column
 */
private[artan] trait HasMeasurementModelCol extends Params {

  /**
   * Param for column name to specify measurement model from input DataFrame rather than
   * a constant measurement model for all filters. Overrides [[measurementModel]] param.
   * @group param
   */
  final val measurementModelCol: Param[String] = new Param[String](
    this,
    "measurementModelCol",
    "Column name for specifying measurement model from input DataFrame rather than" +
    "a constant measurement model for all filters")

  /**
   * Getter for measurement model column
   * @group getParam
   */
  final def getMeasurementModelCol: String = $(measurementModelCol)
}


/**
 * Param for measurement noise column
 */
private[artan] trait HasMeasurementNoiseCol extends Params {

  /**
   * Param for column name to specify measurement noise from input DataFrame rather than
   * a constant measurement noise for all filters. Overrides [[measurementNoise]] param
   */
  final val measurementNoiseCol: Param[String] = new Param[String](
    this,
    "measurementNoiseCol",
    "Column name for specifying measurement noise from input DataFrame rather than" +
    "a constant measurement noise for all filters")

  /**
   * Getter for measurement noise matrix column
   * @group getParam
   */
  final def getMeasurementNoiseCol: String = $(measurementNoiseCol)
}


/**
 * Param for process model column
 */
private[artan] trait HasProcessModelCol extends Params {

  /**
   * Param for column name to specify process model from input DataFrame rather than
   * a constant process model for all filters. Overrides [[processModel]] param.
   * @group param
   */
  final val processModelCol: Param[String] = new Param[String](
    this,
    "processModelCol",
    "Column name for specifying process model from input DataFrame rather than" +
    "a constant process model for all filters")

  /**
   * Getter for process model matrix column.
   * @group getParam
   */
  final def getProcessModelCol: String = $(processModelCol)
}

/**
 * Param for process noise column
 */
private[artan] trait HasProcessNoiseCol extends Params {

  /**
   * Param for column name to specify process noise matrix from input DataFrame rather than
   * a constant process noise for all filters. Overrides [[processNoise]] param
   */
  final val processNoiseCol: Param[String] = new Param[String](
    this,
    "processNoiseCol",
    "Column name for specifying process noise matrix from input DataFrame rather than" +
    "a constant process noise for all filters")

  /**
   * Getter for process noise matrix column.
   * @group getParam
   */
  final def getProcessNoiseCol: String = $(processNoiseCol)
}


/**
 * Param for control column
 */
private[artan] trait HasControlCol extends Params {

  /**
   * Param for column name to specify control vector.
   * @group param
   */
  final val controlCol: Param[String] = new Param[String](
    this,
    "controlCol",
    "Column name for specifying control vector")

  /**
   * Getter for control vector column
   * @group getParam
   */
  final def getControlCol: String = $(controlCol)
}


/**
 * Param for control function column
 */
private[artan] trait HasControlFunctionCol extends Params {

  /**
   * Param for column name to specify control matrix
   * @group param
   */
  final val controlFunctionCol: Param[String] = new Param[String](
    this,
    "controlFunctionCol",
    "Column name for specifying control matrix")

  /**
   * Getter for control function matrix column
   * @group getParam
   */
  final def getControlFunctionCol: String = $(controlFunctionCol)
}


/**
 * Param for enabling mahalanobis calculation.
 */
private[artan] trait HasCalculateMahalanobis extends Params {

  /**
   * Param for enabling mahalanobis calculation. When true, mahalanobis distance of residual will be calculated &
   * added to output DataFrame.
   * Default is false.
   * @group param
   */
  final val calculateMahalanobis: BooleanParam = new BooleanParam(
    this,
    "calculateMahalanobis",
    "When true, mahalanobis distance of residual will be calculated & added to output DataFrame." +
    "Default is false.")

  setDefault(calculateMahalanobis, false)

  /**
   * Getter for mahalanobis calculation flag param
   * @group getParam
   */
  final def getCalculateMahalanobis: Boolean = $(calculateMahalanobis)
}


/**
 * Param for enabling loglikelihood calculation
 */
private[artan] trait HasCalculateLoglikelihood extends Params {

  /**
   * Param for enabling loglikelihood calculation. When true, loglikelihood of residual will be calculated & added
   * to output DataFrame. Default is false.
   * @group param
   */
  final val calculateLoglikelihood: BooleanParam = new BooleanParam(
    this,
    "calculateLoglikelihood",
    "When true, loglikelihood of residual will be calculated & added to output DataFrame. Default is false")

  setDefault(calculateLoglikelihood, false)

  /**
   * Getter for loglikelihood calculation flag param
   * @group getParam
   */
  final def getCalculateLoglikelihood: Boolean = $(calculateLoglikelihood)
}


/**
 * Param for enabling sliding likelihood calculation
 */
private[artan] trait HasCalculateSlidingLikelihood extends Params {

  /**
   * Param for enabling sliding likelihood calculation. When true, sliding likelihood sum of residual will be
   * calculated & added to output DataFrame. Default is false
   * @group param
   */
  final val calculateSlidingLikelihood: BooleanParam = new BooleanParam(
    this,
    "calculateSlidingLikelihood",
    "When true, sliding likelihood sum of residual will be calculated & added to output DataFrame. Default is false")

  setDefault(calculateSlidingLikelihood, false)

  /**
   * Getter for sliding likelihood calculation flag param
   * @group getParam
   */
  final def getCalculateSlidingLikelihood: Boolean = $(calculateSlidingLikelihood)
}

/**
 * Param for sliding likelihood window
 */
private[artan] trait HasSlidingLikelihoodWindow extends Params {

  /**
   * Param for sliding likelihood window. Number of consecutive measurements to include in the total likelihood calculation
   * Default is 1. Should be >=1.
   * @group param
   */
  final val slidingLikelihoodWindow: IntParam = new IntParam(
    this,
    "slidingLikelihoodWindow",
    "Number of consecutive measurements to include in the total likelihood calculation",
    ParamValidators.gtEq(1))

  setDefault(slidingLikelihoodWindow, 1)

  /**
   * Getter for sliding likelihood window param
   * @group getParam
   */
  final def getSlidingLikelihoodWindow: Int = $(slidingLikelihoodWindow)
}

/**
 * Param for mmae measurement window
 */
private[artan] trait HasMultipleModelMeasurementWindowDuration extends Params {

  /**
   * Param for measurement window of MMAE mode. Window duration as string for grouping measurements in same
   * window for MMAE filter aggregation
   * @group param
   */
  final val multipleModelMeasurementWindowDuration: Param[String] = new Param[String](
    this,
    "multipleModelMeasurementWindowDuration",
    "Window duration for grouping measurements in same window for MMAE filter aggregation"
  )

  /**
   * Getter for mmae measurement window
   * @group getParam
   */
  final def getMultipleModelMeasurementWindow: String = $(multipleModelMeasurementWindowDuration)
}

/**
 * Param for enabling output of system matrices
 */
private[artan] trait HasOutputSystemMatrices extends Params {

  /**
   * Param for enabling output of system matrices. When true, system matrices will be also added to output DataFrame.
   * Default is false
   * @group param
   */
  final val outputSystemMatrices: BooleanParam = new BooleanParam(
    this,
    "outputSystemMatrices",
    "When true, system matrices will be also added to output DataFrame. Default is false")

  setDefault(outputSystemMatrices, false)

  /**
   * Getter for system matrices output flag
   * @group getParam
   */
  final def getOutputSystemMatrices: Boolean = $(outputSystemMatrices)
}