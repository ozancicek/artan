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

import com.ozancicek.artan.ml.state.{LMSState, LMSUpdate, LMSOutput}
import com.ozancicek.artan.ml.state.{StateUpdateFunction, StatefulTransformer}
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
  extends StatefulTransformer[String, LMSUpdate, LMSState, LMSOutput]
  with HasGroupKeyCol with HasLabelCol with HasFeaturesCol with HasStateMean {

  implicit val lmsUpdateEncoder = Encoders.product[LMSUpdate]
  implicit val lmsOutputEncoder = Encoders.product[LMSOutput]
  implicit val groupKeyEncoder = Encoders.STRING

  def this(stateSize: Int) = this(stateSize, Identifiable.randomUID("leastMeanSquaresFilter"))

  def keyFunc = (in: LMSUpdate) => in.groupKey

  override def copy(extra: ParamMap): LeastMeanSquaresFilter = defaultCopy(extra)

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol, "label")

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  setDefault(featuresCol, "features")

  def setGroupKeyCol(value: String): this.type = set(groupKeyCol, value)

  private def validateSchema(schema: StructType): Unit = {
    require(isSet(groupKeyCol), "Group key column must be set")
    require(schema($(groupKeyCol)).dataType == StringType, "Group key column must be StringType")
    require(schema($(labelCol)).dataType == DoubleType)
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType)
  }

  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    lmsUpdateEncoder.schema
  }

  def filter(dataset: Dataset[_]): Dataset[LMSOutput] = {
    transformSchema(dataset.schema)
    val lmsUpdateDS = dataset
      .withColumn("groupKey", col($(groupKeyCol)))
      .withColumn("label", col($(labelCol)))
      .withColumn("features", col($(featuresCol)))
      .select("groupKey", "label", "features")
      .as(lmsUpdateEncoder)
    transformWithState(lmsUpdateDS)
  }

  def transform(dataset: Dataset[_]): DataFrame = filter(dataset).toDF

  def stateUpdateFunc = new LeastMeanSquaresUpdateFunction(
    getStateMean)

}


private[ml] class LeastMeanSquaresUpdateFunction(
    val stateMean: Vector)
  extends StateUpdateFunction[String, LMSUpdate, LMSState, LMSOutput] {

  def updateGroupState(
    key: String,
    row: LMSUpdate,
    state: Option[LMSState]): Option[LMSState] = {

    val currentState = state
      .getOrElse(LMSState(key, 0L, stateMean))

    val features = row.features
    val gain = features.copy
    BLAS.scal(1.0/BLAS.dot(features, features), gain)
    val residual = row.label -  BLAS.dot(features, currentState.mean)

    val estMean = currentState.mean.copy
    BLAS.axpy(residual, gain, estMean)
    val newState = LMSState(key, currentState.index + 1, estMean)
    Some(newState)
  }
}
