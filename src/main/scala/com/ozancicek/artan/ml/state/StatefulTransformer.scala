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
import java.sql.Timestamp

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for a stateful [[org.apache.spark.ml.Transformer]]. Performs a stateful transformation
 * using flatMapGroupsWithState function, which is specified with a [[StateUpdateSpec]]. Currently, only
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
  StateType <: State : ClassTag,
  OutType <: KeyedOutput[GroupKeyType] : TypeTag,
  ImplType <: StatefulTransformer[GroupKeyType, RowType, StateType, OutType, ImplType]] extends Transformer
  with HasStateTimeoutMode with HasWatermarkCol with HasWatermarkDuration with HasStateKeyCol[GroupKeyType]
  with HasStateTimeoutDuration {

  def setStateKeyCol(value: String): ImplType = set(stateKeyCol, value).asInstanceOf[ImplType]

  def setStateTimeoutMode(value: String): ImplType = set(timeoutMode, value).asInstanceOf[ImplType]
  setDefault(timeoutMode, "none")

  def setWatermarkCol(value: String): ImplType = set(watermarkCol, value).asInstanceOf[ImplType]

  def setWatermarkDuration(value: String): ImplType = set(watermarkDuration, value).asInstanceOf[ImplType]

  /* Function to pass to flatMapGroupsWithState */
  protected def stateUpdateSpec: StateUpdateSpec[GroupKeyType, RowType, StateType, OutType]

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

    val withStateKey = in.withColumn("stateKey", getStateKeyColumn)

    val toTyped = (df: DataFrame) => df
      .select(rowFields.head, rowFields.tail: _*).as(rowEncoder)

    val inputDS = if (isSet(watermarkCol)) {
      val typed = toTyped(withStateKey.withColumn("eventTime", col($(watermarkCol))))
      typed.withWatermark("eventTime", $(watermarkDuration))
    } else {
      toTyped(withStateKey.withColumn("eventTime", lit(null).cast(TimestampType)))
    }

    val ops = StateSpecOps(getTimeoutMode, getStateTimeoutDuration)

    inputDS.groupByKey(keyFunc)
      .flatMapGroupsWithState(
        OutputMode.Append,
        getTimeoutMode.conf)(stateUpdateSpec.stateFunc(ops))
  }
}


/**
 * Param for state key column
 */
private[state] trait HasStateKeyCol[KeyType] extends Params {

  protected val defaultStateKey: KeyType

  final val stateKeyCol: Param[String] = new Param[String](
    this, "stateKeyCol", "state key column name")

  final def getStateKeyColname: String = $(stateKeyCol)

  final def getStateKeyColumn: Column = if (isSet(stateKeyCol)) col($(stateKeyCol)) else lit(defaultStateKey)
}


/**
 * Param for watermark column name
 */
private[state] trait HasWatermarkCol extends Params {
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
private[state] trait HasWatermarkDuration extends Params {
  final val watermarkDuration: Param[String] = new Param[String](
    this,
    "watermarkDuration",
    "Watermark duration"
  )

  def getWatermarkDuration: String = $(watermarkDuration)
}


/**
 * Param for state timeout duration
 */
private[state] trait HasStateTimeoutDuration extends Params {
  final val stateTimeoutDuration: Param[String] = new Param[String](
    this,
    "stateTimeoutDuration",
    "State timeout duration"
  )

  def getStateTimeoutDuration: Option[String] = {
    if (isSet(stateTimeoutDuration)) Some($(stateTimeoutDuration)) else None
  }
}


/**
 * Param for timeout mode
 */
private[state] trait HasStateTimeoutMode extends Params {

  private val supportedTimeoutMods = Set("none", "process", "event")

  final val timeoutMode: Param[String] = new Param[String](
    this,
    "timeoutMode",
    "Group state timeout mode. Supported options:" +
    s"${supportedTimeoutMods.mkString(", ")} . (Default none)"
  )

  def getTimeoutMode: TimeoutMode = $(timeoutMode) match {
    case "none" => NoTimeout
    case "process" => ProcessingTimeTimeout
    case "event" => EventTimeTimeout
    case _ => throw new Exception("Unsupported mode")
  }

}


/**
 *  Enumeration for timeout mode
 */
private[state] sealed trait TimeoutMode {
  def conf: GroupStateTimeout
}


private[state] case object NoTimeout extends TimeoutMode {
  def conf: GroupStateTimeout = GroupStateTimeout.NoTimeout
}


private[state] case object ProcessingTimeTimeout extends TimeoutMode {
  def conf: GroupStateTimeout = GroupStateTimeout.ProcessingTimeTimeout
}


private[state] case object EventTimeTimeout extends TimeoutMode {
  def conf: GroupStateTimeout = GroupStateTimeout.EventTimeTimeout
}


private[state] case class StateSpecOps(
    timeoutMode: TimeoutMode,
    timeoutDuration: Option[String])

/**
 * Base spec for creating function to be used in flatMapGroupsWithState. Performs a stateful transformation
 * from [[RowType]] to [[OutType]], while storing [[StateType]] for each group denoted by key [[GroupKeyType]].
 *
 * @tparam GroupKeyType Key type of groups.
 * @tparam RowType Input type
 * @tparam StateType State type
 * @tparam OutType Output type
 */
private[ml] trait StateUpdateSpec[
  GroupKeyType,
  RowType <: KeyedInput[GroupKeyType],
  StateType <: State,
  OutType <: KeyedOutput[GroupKeyType]]
  extends Serializable {

  protected def stateToOutput(
    key: GroupKeyType,
    row: RowType,
    state: StateType): OutType

  protected def updateGroupState(
    key: GroupKeyType,
    row: RowType,
    state: Option[StateType]): Option[StateType]


  private implicit def orderedIfSet: Ordering[Option[Timestamp]] = new Ordering[Option[Timestamp]] {
    def compare(left: Option[Timestamp], right: Option[Timestamp]): Int = (left, right) match {
      case (Some(l), Some(r)) => l.compareTo(r)
      case (None, Some(r)) => -1
      case (Some(l), None) => 1
      case (None, None) => 0
    }
  }

  private def setStateTimeout(
    groupState: GroupState[StateType],
    eventTime: Option[Timestamp],
    ops: StateSpecOps): Unit = {
    (ops.timeoutMode, ops.timeoutDuration) match {
      case (EventTimeTimeout, Some(duration))=> eventTime
        .foreach(ts => groupState.setTimeoutTimestamp(ts.getTime, duration))
      case (ProcessingTimeTimeout, Some(duration)) => groupState.setTimeoutDuration(duration)
      case _ =>
    }
  }

  def stateFunc(ops: StateSpecOps): (GroupKeyType, Iterator[RowType], GroupState[StateType]) => Iterator[OutType] = {
    (key: GroupKeyType, rows: Iterator[RowType], groupState: GroupState[StateType]) => {

      if (groupState.hasTimedOut) {
        groupState.remove()
      }

      /* Need access to previous state while creating an output from measurement. Gather output in a queue
      while keeping the state in a pair*/
      val outputQueue = Queue[Option[OutType]]()
      val statePair = (groupState.getOption, groupState.getOption)

      /* If state times out input rows will be empty, resulting in empty output iterator */
      rows.toSeq.sortBy(_.eventTime).foldLeft((outputQueue, statePair)) {
        case ((q, (_, currentState)), row) => {
          // Calculate the next state
          val nextState = updateGroupState(key, row, currentState)

          // Update the state and set timeout if Some(state)
          nextState.foreach { s =>
            groupState.update(s)
            setStateTimeout(groupState, row.eventTime, ops)
          }

          // Convert state to out type, push to output queue
          val out = nextState.map(s => stateToOutput(key, row, s))
          (q :+ out, (currentState, nextState))
        }
      }._1.flatten.toIterator
    }
  }

}