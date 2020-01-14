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

package com.ozancicek.artan.ml.state

import org.apache.spark.ml.Transformer
import org.apache.spark.sql.streaming.{GroupState, GroupStateTimeout, OutputMode}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, lit}

import scala.collection.immutable.Queue
import scala.reflect.ClassTag
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.TimestampType

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for a stateful [[org.apache.spark.ml.Transformer]]. Performs a stateful transformation
 * using flatMapGroupsWithState function, which is specified with a [[StateUpdateFunction]]. Currently, only
 * append output mode is supported.
 *
 * @tparam GroupKeyType Key type of groups.
 * @tparam RowType Input type
 * @tparam StateType State type
 * @tparam OutType Output type
 * @tparam ImplType Implementing class type
 */
private[ml] abstract class StatefulTransformer[
  GroupKeyType,
  RowType <: KeyedInput[GroupKeyType] : TypeTag : Manifest,
  StateType <: KeyedState[GroupKeyType, RowType, OutType] : ClassTag,
  OutType <: Product : TypeTag,
  ImplType <: StatefulTransformer[GroupKeyType, RowType, StateType, OutType, ImplType]] extends Transformer
  with HasStateTimeoutMode with HasWatermarkCol with HasWatermarkDuration with HasStateKeyCol {

  def setStateKeyCol(value: String): ImplType = set(stateKeyCol, value).asInstanceOf[ImplType]

  def setStateTimeoutMode(value: String): ImplType = set(timeoutMode, value).asInstanceOf[ImplType]
  setDefault(timeoutMode, "none")

  def setWatermarkCol(value: String): ImplType = set(watermarkCol, value).asInstanceOf[ImplType]

  def setWatermarkDuration(value: String): ImplType = set(watermarkDuration, value).asInstanceOf[ImplType]

  /* Function to pass to flatMapGroupsWithState */
  protected def stateUpdateFunc: StateUpdateFunction[GroupKeyType, RowType, StateType, OutType]

  /* Keying function for groupByKey*/
  protected def keyFunc: RowType => GroupKeyType = (in: RowType) => in.stateKey

  /* State is encoded with kyro in order to support schema evolution. Others are encoded with spark encoders. */
  protected implicit val stateEncoder = Encoders.kryo[StateType]
  protected implicit val rowEncoder = Encoders.product[RowType]
  protected implicit val outEncoder = Encoders.product[OutType]

  /* Get input case class fields with reflection*/
  private def rowFields: List[String] = implicitly[Manifest[RowType]]
    .runtimeClass.getDeclaredFields.map(_.getName).toList

  protected def transformWithState(
    in: DataFrame)(
    implicit keyEncoder: Encoder[GroupKeyType]): Dataset[OutType] = {

    val withStateKey = in.withColumn("stateKey", col(getStateKeyCol))

    val toTyped = (df: DataFrame) => df
      .select(rowFields.head, rowFields.tail: _*).as(rowEncoder)

    val inputDS = if (isSet(watermarkCol)) {
      val typed = toTyped(withStateKey.withColumn("eventTime", col($(watermarkCol))))
      typed.withWatermark("eventTime", $(watermarkDuration))
    } else {
      toTyped(withStateKey.withColumn("eventTime", lit(null).cast(TimestampType)))
    }

    inputDS.groupByKey(keyFunc)
      .flatMapGroupsWithState(
        OutputMode.Append,
        getTimeoutConf)(stateUpdateFunc)
  }
}


/**
 * Param for state key column
 */
trait HasStateKeyCol extends Params {

  final val stateKeyCol: Param[String] = new Param[String](
    this, "stateKeyCol", "state key column name")

  final def getStateKeyCol: String = $(stateKeyCol)
}


/**
 * Param for watermark column name
 */
trait HasWatermarkCol extends Params {
  final val watermarkCol: Param[String] = new Param[String](
    this,
    "watermarkCol",
    "Watermark column name for event time"
  )

  def getWatermarkCol: String = $(watermarkCol)
}


/**
 * Param for watermark duration
 */
trait HasWatermarkDuration extends Params {
  final val watermarkDuration: Param[String] = new Param[String](
    this,
    "watermarkDuration",
    "Watermark duration"
  )

  def getWatermarkDuration: String = $(watermarkDuration)
}


trait HasStateTimeoutMode extends Params {

  private val supportedTimeoutMods = Set("none", "process", "event")

  final val timeoutMode: Param[String] = new Param[String](
    this,
    "timeoutMode",
    "Group state timeout mode. Supported options:" +
    s"${supportedTimeoutMods.mkString(", ")} . (Default none)"
  )

  def getTimeoutMode: String = $(timeoutMode)

  def getTimeoutConf: GroupStateTimeout = getTimeoutMode match {
    case "none" => GroupStateTimeout.NoTimeout()
    case "process" => GroupStateTimeout.ProcessingTimeTimeout()
    case "event" => GroupStateTimeout.EventTimeTimeout()
    case _ => throw new Exception("Unsupported mode")
  }
}

/**
 * Base trait for a function to be used in flatMapGroupsWithState. Performs a stateful transformation
 * from [[RowType]] to [[OutType]], while storing [[StateType]] for each group denoted by key [[GroupKeyType]].
 *
 * @tparam GroupKeyType Key type of groups.
 * @tparam RowType Input type
 * @tparam StateType State type
 * @tparam OutType Output type
 */
private[ml] trait StateUpdateFunction[
  GroupKeyType,
  RowType <: Product,
  StateType <: KeyedState[GroupKeyType, RowType, OutType],
  OutType <: Product] extends Function3[GroupKeyType, Iterator[RowType], GroupState[StateType], Iterator[OutType]]
  with Serializable {

  protected def updateGroupState(
    key: GroupKeyType,
    row: RowType,
    state: Option[StateType]): Option[StateType]

  def apply(
    key: GroupKeyType,
    rows: Iterator[RowType],
    groupState: GroupState[StateType]
  ): Iterator[OutType] = {

    if (groupState.hasTimedOut) {
      groupState.remove()
    }

    val outputQueue = Queue[Option[OutType]]()
    val statePair = (groupState.getOption, groupState.getOption)

    rows.foldLeft((outputQueue, statePair)) {
      case ((q, (_, currentState)), row) => {
        val nextState = updateGroupState(key, row, currentState)
        nextState.foreach(s => groupState.update(s))
        (q :+ nextState.map(_.asOut(row)), (currentState, nextState))
      }
    }._1.flatten.toIterator
  }
}