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

import com.ozancicek.artan.ml.state.{LMSState, LMSInput, LMSOutput}
import com.ozancicek.artan.ml.state.{StateUpdateSpec, StatefulTransformer}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{BLAS}
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._


class LeastMeanSquaresFilter(
    val stateSize: Int,
    override val uid: String)
  extends StatefulTransformer[String, LMSInput, LMSState, LMSOutput, LeastMeanSquaresFilter]
  with HasLabelCol with HasFeaturesCol with HasInitialState {

  implicit val stateKeyEncoder = Encoders.STRING

  def this(stateSize: Int) = this(stateSize, Identifiable.randomUID("leastMeanSquaresFilter"))

  protected val defaultStateKey: String = "filter.leastMeanSquaresFilter"

  override def copy(extra: ParamMap): LeastMeanSquaresFilter = defaultCopy(extra)

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol, "label")

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  setDefault(featuresCol, "features")

  private def validateSchema(schema: StructType): Unit = {
    validateWatermarkColumns(schema)
    if (isSet(stateKeyCol)) {
      require(schema($(stateKeyCol)).dataType == StringType, "Group key column must be StringType")
    }
    require(schema($(labelCol)).dataType == DoubleType)
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType)
  }

  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    outEncoder.schema
  }

  def filter(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    val lmsUpdateDS = dataset
      .withColumn("label", col($(labelCol)))
      .withColumn("features", col($(featuresCol)))
    transformWithState(lmsUpdateDS)
  }

  def transform(dataset: Dataset[_]): DataFrame = filter(dataset)

  protected def stateUpdateSpec: LeastMeanSquaresUpdateSpec = new LeastMeanSquaresUpdateSpec(
    getInitialState)

}


private[filter] class LeastMeanSquaresUpdateSpec(
    val stateMean: Vector)
  extends StateUpdateSpec[String, LMSInput, LMSState, LMSOutput] {

  protected def stateToOutput(key: String, row: LMSInput, state: LMSState): LMSOutput = {
    LMSOutput(
      key,
      state.stateIndex,
      state.state,
      row.eventTime)
  }

  def updateGroupState(
    key: String,
    row: LMSInput,
    state: Option[LMSState]): Option[LMSState] = {

    val currentState = state
      .getOrElse(LMSState(0L, stateMean))

    val features = row.features
    val gain = features.copy
    BLAS.scal(1.0/BLAS.dot(features, features), gain)
    val residual = row.label -  BLAS.dot(features, currentState.state)

    val estMean = currentState.state.copy
    BLAS.axpy(residual, gain, estMean)
    val newState = LMSState(currentState.stateIndex + 1, estMean)
    Some(newState)
  }
}
