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

package com.github.ozancicek.artan.ml.state

import org.apache.spark.ml.linalg.{Matrix, Vector}

import scala.collection.immutable.Queue
import java.sql.Timestamp

import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution

/**
 * Case class for inputs of a kalman filter. Let state at step k be denoted with x_k, measurement with
 * z_k, where x_k and z_k are vectors of length n_state and n_obs
 *
 * State evolution:
 * x_k = F_k * x_k-1 + B_k * u_k + w_k
 *
 * Measurement:
 * z_k = H_k * x_k + v_k
 *
 * @param stateKey Key of the filter state.
 * @param initialState x_0, Vector with length n_state
 * @param measurement z_k, Vector with length n_obs
 * @param measurementModel H_k, Matrix with dimensions (n_state, n_obs)
 * @param measurementNoise v_k, Matrix with dimensions (n_obs, n_obs)
 * @param processModel F_k, Matrix with dimensions (n_state, n_state)
 * @param processNoise w_k, Matrix with dimensions (n_state, n_state)
 * @param control u_k, Vector with length n_state
 * @param controlFunction B_k, Matrix with dimensions (n_state, n_state)
 * @param eventTime event time of input
 */
private[ml] case class KalmanInput(
    stateKey: String,
    initialState: MultivariateGaussianDistribution,
    measurement: Option[Vector],
    measurementModel: Option[Matrix],
    measurementNoise: Option[Matrix],
    processModel: Option[Matrix],
    processNoise: Option[Matrix],
    control: Option[Vector],
    controlFunction: Option[Matrix],
    eventTime: Option[Timestamp]) extends KeyedInput[String]



/**
 * Case class for representing the output state of a kalman filter.
 * Let state at step k be denoted with x_k, measurement with
 * z_k, where x_k and z_k are vectors of lenght n_state and n_obs
 *
 *
 * @param stateKey Key of the state
 * @param stateIndex index of the filter, incremented only on state evolution
 * @param state x_k, the state vector with length n_state
 * @param residual residual of x_k and z_k, vector with length n_obs
 * @param eventTime event time of input measurement
 * @param processModel process model used at this state
 * @param processNoise process noise used at this state
 * @param measurementModel measurement model used at this state
 * @param slidingLikelihood likelihood of consecutive measurements in a sliding window at this state
 */
case class KalmanOutput(
    stateKey: String,
    stateIndex: Long,
    stepIndex: Long,
    state: MultivariateGaussianDistribution,
    residual: Option[MultivariateGaussianDistribution],
    eventTime: Option[Timestamp],
    processModel: Option[Matrix],
    processNoise: Option[Matrix],
    measurement: Option[Vector],
    measurementModel: Option[Matrix],
    slidingLikelihood: Option[Double]) extends KeyedOutput[String]

/**
 * Internal representation of the state of a kalman filter.
 */
private[ml] case class KalmanState(
    stateIndex: Long,
    state: MultivariateGaussianDistribution,
    residual: Option[MultivariateGaussianDistribution],
    processNoise: Option[Matrix],
    slidingLoglikelihood: Queue[Double]) extends State


/**
 * Case class representing output state of kalman RTS smoother
 * @param stateKey Key of the state
 * @param stateIndex index of the filter, incremented only on state evolution
 * @param state x_k, the state vector with length n_state
 * @param rtsGain Smoother gain calculated from RTS Smoother
 * @param eventTime event time of input measurement
 */
case class RTSOutput(
    stateKey: String,
    stateIndex: Long,
    stepIndex: Long,
    state: MultivariateGaussianDistribution,
    rtsGain: Matrix,
    eventTime: Option[Timestamp]) extends KeyedOutput[String]


private[ml] case class MMAEState(
    loglikelihood: Queue[Double],
    stateIndex: Long,
    state: MultivariateGaussianDistribution,
    eventTime: Option[Timestamp]) extends State