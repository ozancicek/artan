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

package com.ozancicek.artan.ml.smoother

import com.ozancicek.artan.ml.linalg.LinalgUtils
import com.ozancicek.artan.ml.state.{KalmanOutput, RTSOutput, StateUpdateSpec, StatefulTransformer}
import org.apache.spark.ml.BLAS
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}
import org.apache.spark.sql.types.StructType
import com.ozancicek.artan.ml.filter._

import scala.collection.immutable.Queue

/**
 * Fixed lag linear kalman smoother.
 *
 * @param stateSize size of the state vector
 * @param measurementSize size of the measurement vector
 */
class LinearKalmanSmoother(
    val stateSize: Int,
    val measurementSize: Int,
    override val uid: String)
  extends StatefulTransformer[String, KalmanOutput, Queue[KalmanOutput], RTSOutput, LinearKalmanSmoother]
  with KalmanUpdateParams with HasInitialCovariance with HasInitialState with HasFadingFactor {

  def this(stateSize: Int, measurementSize: Int) = {
    this(stateSize, measurementSize, Identifiable.randomUID("LinearKalmanSmoother"))
  }
  implicit val stateKeyEncoder = Encoders.STRING
  protected val defaultStateKey: String = "smoother.LinearKalmanSmoother"

  final val fixedLag: IntParam = new IntParam(
    this,
    "fixedLag",
    "fixed lag param",
    ParamValidators.gt(1))
  setDefault(fixedLag, 2)

  def setFixedLag(value: Int): this.type = set(fixedLag, value)
  def getFixedLag: Int = $(fixedLag)

  def stateUpdateSpec: LinearKalmanSmootherSpec = new LinearKalmanSmootherSpec(getFixedLag)

  def transformSchema(schema: StructType): StructType = {
    outEncoder.schema
  }

  override def copy(extra: ParamMap): LinearKalmanSmoother = defaultCopy(extra)

  def transform(dataset: Dataset[_]): DataFrame = {
    transformWithState(dataset.toDF)
  }
}


private[smoother] class LinearKalmanSmootherSpec(lag: Int)
  extends StateUpdateSpec[String, KalmanOutput, Queue[KalmanOutput], RTSOutput] {

  private def updateRTSOutput(head: Option[RTSOutput], in: KalmanOutput): RTSOutput = {
    head match {
      case None => {
        // First measurement, only calculate one lagged covariance

        val measurementModel = in.measurementModel.get.toDense

        // ident = I - K * H
        val ident = DenseMatrix.eye(in.state.size)
        BLAS.gemm(-1.0, in.gain.get, measurementModel, 1.0, ident)
        val laggedCov = ident
          .multiply(in.processModel.get.toDense)
          .multiply(in.stateCovariance.toDense)

        RTSOutput(
          in.stateKey,
          in.stateIndex,
          in.state,
          in.stateCovariance,
          DenseMatrix.zeros(in.state.size, in.state.size),
          laggedCov,
          in.eventTime)
      }
      case Some(prev) => {
        val model = in.processModel.get.toDense
        val nextState = model.multiply(in.state)
        val covUpdate = model.multiply(in.stateCovariance.toDense)

        val nextCov = in.processNoise.get.toDense
        BLAS.gemm(1.0, covUpdate, model.transpose, 1.0, nextCov)
        val gain = in.stateCovariance.multiply(model.transpose).multiply(LinalgUtils.pinv(nextCov))

        val residual = prev.state.copy
        BLAS.axpy(-1.0, nextState, residual)

        val newMean = in.state.toDense.copy
        BLAS.gemv(1.0, gain, residual, 1.0, newMean)

        val covDiff = prev.stateCovariance.toDense.copy
        BLAS.axpy(-1.0, nextCov, covDiff)
        val newCow = in.stateCovariance.toDense.copy
        BLAS.gemm(1.0, gain.multiply(covDiff), gain.transpose, 1.0, newCow)

        val laggedCovDiff = prev.laggedStateCovariance.toDense.copy
        BLAS.axpy(-1.0, covUpdate, laggedCovDiff)

        val laggedCov = in.stateCovariance.multiply(prev.rtsGain.toDense.transpose)
        BLAS.gemm(1.0, gain.multiply(laggedCovDiff), prev.rtsGain.toDense.transpose, 1.0, laggedCov)

        RTSOutput(
          in.stateKey,
          in.stateIndex,
          newMean,
          newCow,
          gain,
          laggedCov,
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
      val (head, tail) = state.reverse.dequeue
      val headOutput = updateRTSOutput(None, head)

      tail.foldLeft(List(headOutput)) {
        case(x::xs, in) => updateRTSOutput(Some(x), in)::x::xs
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

    // State is just fixed size queue of KalmanOutput
    if (currentState.size == lag) {
      Some(currentState.dequeue._2.enqueue(row))
    }
    else {
      Some(currentState.enqueue(row))
    }
  }
}