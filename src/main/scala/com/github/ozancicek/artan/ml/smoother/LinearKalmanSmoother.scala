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

package com.github.ozancicek.artan.ml.smoother

import com.github.ozancicek.artan.ml.filter.{KalmanUpdateParams, LinearKalmanFilter}
import com.github.ozancicek.artan.ml.linalg.{LinalgOptions, LinalgUtils}
import com.github.ozancicek.artan.ml.state.{KalmanOutput, RTSOutput, StateUpdateSpec, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution
import org.apache.spark.ml.BLAS
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}
import org.apache.spark.sql.types.StructType

import scala.collection.immutable.Queue

/**
 * Fixed lag linear kalman smoother using Rauch-Tung-Striebel method. The smoother is implemented with a stateful
 * spark transformer for running parallel smoothers /w spark dataframes.
 * Transforms an input dataframe of noisy measurements to dataframe of state estimates using
 * stateful spark transformations, which can be used in both streaming and batch applications.
 *
 * At a time step k and a fixed lag N, the fixed lag linear kalman smoother computes the state estimates of a linear
 * kalman filter based on all measurements made between step k and step k-t. For each time step k >= N, the smoother
 * outputs an estimate for all the time steps between k and k-N. When k < N, the smoother doesn't output any estimates.
 * As a result, the memory requirements of this filter is N times of a linear kalman filter. Since the smoother
 * outputs multiple estimates for a single measurement, it is advised to set event time column
 * of the measurements with setEventTimeCol.
 *
 * Except fixedLag parameter, LinearKalmanSmoother has the same parameters with LinearKalmanFilter
 *
 * @param stateSize size of the state vector
 * @param measurementSize size of the measurement vector
 */
class LinearKalmanSmoother(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends StatefulTransformer[String, KalmanOutput, Queue[KalmanOutput], RTSOutput, LinearKalmanSmoother]
  with KalmanUpdateParams[LinearKalmanSmoother] {

  def this(stateSize: Int, measurementSize: Int) = {
    this(stateSize, measurementSize, Identifiable.randomUID("LinearKalmanSmoother"))
  }

  implicit val stateKeyEncoder = Encoders.STRING
  protected val defaultStateKey: String = "smoother.LinearKalmanSmoother.defaultStateKey"

  final val fixedLag: IntParam = new IntParam(
    this,
    "fixedLag",
    "fixed lag param",
    ParamValidators.gt(1))
  setDefault(fixedLag, 2)

  /**
   * Sets the smoother fixed lag.
   *
   * Default is 2.
   */
  def setFixedLag(value: Int): this.type = set(fixedLag, value)

  def getFixedLag: Int = $(fixedLag)

  protected def stateUpdateSpec: LKFSmootherStateSpec = new LKFSmootherStateSpec(getFixedLag, getLinalgOptions)

  def transformSchema(schema: StructType): StructType = {
    asDataFrameTransformSchema(outEncoder.schema)
  }

  override def copy(extra: ParamMap): LinearKalmanSmoother =  {
    val that = new LinearKalmanSmoother(stateSize, measurementSize)
    copyValues(that, extra)
  }

  def transform(dataset: Dataset[_]): DataFrame = {
    val lkf = new LinearKalmanFilter(stateSize, measurementSize)

    val copied = copyValues(lkf, extractParamMap)

    val filtered = copied.filter(dataset).toDF
    asDataFrame(transformWithState(filtered))
  }
}

/**
 * Function spec for calculating LinearKalmanSmoother output from the state. The state is KalmanOutput
 * queue resulting from the LinearKalmanFilter, and the state is transformed to output by RTS method for each
 * update.
 *
 * @param lag fixed size lag.
 */
private[smoother] class LKFSmootherStateSpec(val lag: Int, val ops: LinalgOptions)
  extends StateUpdateSpec[String, KalmanOutput, Queue[KalmanOutput], RTSOutput]{

  /**
   * Function for outputting smoothed output from filtered state. Smoothed output calculation at time step k depends
   * on the filtered output at time step k and smoothed output at previous step.
   *
   * @param head Smoothed output at previous time step. Should be None for initial time step, and Some(value)
   *             for next steps
   * @param in Filtered output at current time step
   * @return RTSOutput, smoothed output at current time step
   */
  private def updateRTSOutput(head: Option[RTSOutput], in: KalmanOutput, stepIndex: Long): RTSOutput = {
    head match {
      case None => {
        // Initial time step logic
        RTSOutput(
          in.stateKey,
          in.stateIndex,
          stepIndex,
          in.state,
          DenseMatrix.zeros(in.state.mean.size, in.state.mean.size),
          in.eventTime)
      }

      case Some(prev) => {
        val model = in.processModel.get.toDense
        val nextState = model.multiply(in.state.mean)
        val covUpdate = model.multiply(in.state.covariance.toDense)

        val nextCov = in.processNoise.get.toDense
        BLAS.gemm(1.0, covUpdate, model.transpose, 1.0, nextCov)
        val gain = in.state.covariance.multiply(model.transpose).multiply(LinalgUtils.pinv(nextCov)(ops))

        val residual = prev.state.mean.copy
        BLAS.axpy(-1.0, nextState, residual)

        val newMean = in.state.mean.toDense.copy
        BLAS.gemv(1.0, gain, residual, 1.0, newMean)

        val covDiff = prev.state.covariance.toDense.copy
        BLAS.axpy(-1.0, nextCov, covDiff)
        val newCov = in.state.covariance.toDense.copy
        BLAS.gemm(1.0, gain.multiply(covDiff), gain.transpose, 1.0, newCov)

        RTSOutput(
          in.stateKey,
          in.stateIndex,
          stepIndex,
          MultivariateGaussianDistribution(newMean, newCov),
          gain,
          in.eventTime
        )
      }
    }
  }

  protected def stateToOutput(
    key: String,
    row: KalmanOutput,
    state: Queue[KalmanOutput]): List[RTSOutput] = {
    if (state.size == lag) {
      // Start processing from the head of the queue
      val (head, tail) = state.reverse.dequeue
      // Generate the initial smoothed output
      val headOutput = updateRTSOutput(None, head, 0L)

      // Rest of the output is generated with fold logic, as output at current time step depends on the output
      // at previous time step.
      tail.zipWithIndex.foldLeft(List(headOutput)) {
        case(x::xs, (currentState, ind)) => updateRTSOutput(Some(x), currentState, ind + 1)::x::xs
        case(Nil, _) => List.empty[RTSOutput]
      }
    }
    else {
      List.empty[RTSOutput]
    }
  }

  def updateGroupState(
    key: String,
    row: KalmanOutput,
    state: Option[Queue[KalmanOutput]]): Option[Queue[KalmanOutput]] = {

    val currentState = state
      .getOrElse(Queue.empty[KalmanOutput])

    // State is KalmanOutput queue with fixed size = lag
    if (currentState.size == lag) {
      Some(currentState.dequeue._2.enqueue(row))
    }
    else {
      Some(currentState.enqueue(row))
    }
  }
}