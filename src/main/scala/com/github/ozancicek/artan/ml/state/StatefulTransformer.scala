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

import org.apache.spark.ml.Transformer
import org.apache.spark.sql.streaming.{GroupState, GroupStateTimeout, OutputMode}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, lit}

import scala.collection.immutable.Queue
import scala.reflect.ClassTag
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.{StructField, StructType, TimestampType}
import java.sql.Timestamp

import scala.reflect.runtime.universe.TypeTag


private[state] trait StatefulTransformerParams[
  ImplType,
  GroupKeyType] extends Params with HasStateTimeoutMode with HasEventTimeCol
  with HasWatermarkDuration with HasStateKeyCol[GroupKeyType] with HasStateTimeoutDuration {

  /**
   * Sets the state key column. Each value in the column should uniquely identify a stateful transformer. Each
   * unique value will result in a separate state.
   * @group setParam
   */
  def setStateKeyCol(value: String): ImplType = set(stateKeyCol, value).asInstanceOf[ImplType]

  /**
   * Sets the state timeout mode. Supported values are 'none', 'process' and 'event'. Enabling state timeout will
   * clear the state after a certain timeout duration which can be set. If a state receives measurements after
   * it times out, the state will be initialized as if it received no measurements.
   *
   * - 'none': No state timeout, state is kept indefinitely.
   *
   * - 'process': Process time based state timeout, state will be cleared if no measurements are received for
   * a duration based on processing time. Effects all states. Timeout duration must be set with
   * setStateTimeoutDuration.
   *
   * - 'event': Event time based state timeout, state will be cleared if no measurements are recieved for a duration
   * based on event time determined by watermark. Effects all states. Timeout duration must be set with
   * setStateTimeoutDuration. Additionally, event time column and it's watermark duration must be set with
   * setEventTimeCol and setWatermarkDuration. Note that this will result in dropping measurements occuring later
   * than the watermark.
   *
   * Default is 'none'
   * @group setParam
   */
  def setStateTimeoutMode(value: String): ImplType = set(timeoutMode, value).asInstanceOf[ImplType]
  setDefault(timeoutMode, "none")

  /**
   * Sets the state timeout duration for all states, only valid when state timeout mode is not 'none'.
   * Must be a valid duration string, such as '10 minutes'.
   * @group setParam
   */
  def setStateTimeoutDuration(value: String): ImplType = set(stateTimeoutDuration, value).asInstanceOf[ImplType]

  /**
   * Sets the event time column in the input DataFrame for event time based state timeout.
   * @group setParam
   */
  def setEventTimeCol(value: String): ImplType = set(eventTimeCol, value).asInstanceOf[ImplType]

  /**
   * Set the watermark duration for all states, only valid when state timeout mode is 'event'.
   * Must be a valid duration string, such as '10 minutes'.
   * @group setParam
   */
  def setWatermarkDuration(value: String): ImplType = set(watermarkDuration, value).asInstanceOf[ImplType]
}

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
  StateType : ClassTag,
  OutType <: KeyedOutput[GroupKeyType] : TypeTag,
  ImplType <: StatefulTransformer[GroupKeyType, RowType, StateType, OutType, ImplType]] extends Transformer
  with StatefulTransformerParams[ImplType, GroupKeyType] {

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

  protected def validateWatermarkColumns(schema: StructType): Unit = {
    if (isSet(watermarkDuration)) {
      require(isSet(eventTimeCol), "Event time column must be set when when watermark duration is set")
    }
    if (isSet(eventTimeCol)) {
      require(schema($(eventTimeCol)).dataType == TimestampType)
    }
  }

  protected def asDataFrame(in: Dataset[OutType]): DataFrame = {
    val df = in.toDF
    val withEventTime = if (isSet(eventTimeCol)) {
      df.withColumnRenamed("eventTime", getEventTimeCol)
    }
    else {
      df.drop("eventTime")
    }

    val withWatermark = if (isSet(watermarkDuration)) {
      withEventTime.withWatermark(getEventTimeCol, $(watermarkDuration))
    }
    else {
      withEventTime
    }
    withWatermark
  }

  protected def asDataFrameTransformSchema(schema: StructType): StructType = {
    val filtered = schema.filter(f => f.name == "eventTime")
    if (isSet(eventTimeCol)) {
      StructType(filtered :+ StructField(getEventTimeCol, TimestampType, nullable = true))
    }
    else {
      StructType(filtered)
    }
  }

  protected def transformWithState(
    in: DataFrame)(
    implicit keyEncoder: Encoder[GroupKeyType]): Dataset[OutType] = {

    val withStateKey = in.withColumn("stateKey", getStateKeyColumn)

    val toTyped = (df: DataFrame) => df
      .select(rowFields.head, rowFields.tail: _*).as(rowEncoder)

    val withEventTime = if (isSet(eventTimeCol)) {
      toTyped(withStateKey.withColumn("eventTime", col($(eventTimeCol))))
    }
    else {
      toTyped(withStateKey.withColumn("eventTime", lit(null).cast(TimestampType)))
    }

    val withWatermark = if (isSet(watermarkDuration)) {
      withEventTime.withWatermark("eventTime", $(watermarkDuration))
    }
    else {
      withEventTime
    }

    val ops = StateSpecOps(getTimeoutMode, getStateTimeoutDuration)

    val outDS = withWatermark.groupByKey(keyFunc)
      .flatMapGroupsWithState(
        OutputMode.Append,
        getTimeoutMode.conf)(stateUpdateSpec.stateFunc(ops))
    outDS
  }
}


/**
 * Param for state key column
 */
private[state] trait HasStateKeyCol[KeyType] extends Params {

  protected val defaultStateKey: KeyType

  /**
   * Param for state key column. State keys uniquely identify the each state in stateful transformers,
   * thus controlling the number of states and the degree of parallelization"
   * @group param
   */
  final val stateKeyCol: Param[String] = new Param[String](
    this, "stateKeyCol",
    "State key column. State keys uniquely identify the each state in stateful transformers," +
      "thus controlling the number of states and the degree of parallelization")

  /**
   * Getter for state key column name parameter
   * @group getParam
   */
  final def getStateKeyColname: String = $(stateKeyCol)

  /**
   * Getter for state key column
   * @group getParam
   */
  final def getStateKeyColumn: Column = if (isSet(stateKeyCol)) col($(stateKeyCol)) else lit(defaultStateKey)
}


/**
 * Param for event time column name
 */
private[state] trait HasEventTimeCol extends Params {

  /**
   * Param for event time column name, which marks the event time of the received measurements. If set,
   * the measurements will be processed in ascending order according to event time.
   * @group param
   */
  final val eventTimeCol: Param[String] = new Param[String](
    this,
    "eventTimeCol",
    "Column marking the event time of the received measurements" +
    "If set, the measurements will be processed in ascending order according to event time."
  )

  /**
   * Getter for event time column parameter
   * @group getParam
   */
  def getEventTimeCol: String = $(eventTimeCol)
}


/**
 * Param for watermark duration
 */
private[state] trait HasWatermarkDuration extends Params {

  /**
   * Param for watermark duration as string, measured from the [[eventTimeCol]] column. If set, measurements will
   * be processed in append mode with the specified watermark duration.
   * @group param
   */
  final val watermarkDuration: Param[String] = new Param[String](
    this,
    "watermarkDuration",
    "Watermark duration measured from the event time column. If set, measurements will" +
      "be processed in append mode with the specified watermark duration."
  )

  /**
   * Getter for watermark duration parameter
   * @group getParam
   */
  def getWatermarkDuration: String = $(watermarkDuration)
}


/**
 * Param for state timeout duration
 */
private[state] trait HasStateTimeoutDuration extends Params {

  /**
   * Param for state timeout duration.
   */
  final val stateTimeoutDuration: Param[String] = new Param[String](
    this,
    "stateTimeoutDuration",
    "State timeout duration"
  )

  /**
   * Getter for state timeout duration parameter
   * @group getParam
   */
  def getStateTimeoutDuration: Option[String] = {
    if (isSet(stateTimeoutDuration)) Some($(stateTimeoutDuration)) else None
  }
}


/**
 * Param for timeout mode
 */
private[state] trait HasStateTimeoutMode extends Params {

  private val supportedTimeoutMods = Set("none", "process", "event")

  /**
   * Param for timeout mode, controlling the eviction of states which receive no measurement for a certain duration
   * @group param
   */
  final val timeoutMode: Param[String] = new Param[String](
    this,
    "timeoutMode",
    "Group state timeout mode, to control the eviction of states which receive no measurement for a certain duration." +
      "Supported options:" +
    s"${supportedTimeoutMods.mkString(", ")} . (Default none)"
  )

  /**
   * Getter for timeout mode
   * @group getParam
   * @return TimeoutMode
   */
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
  StateType,
  OutType <: KeyedOutput[GroupKeyType]]
  extends Serializable {

  protected def stateToOutput(
    key: GroupKeyType,
    row: RowType,
    state: StateType): List[OutType]

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
      val outputQueue = Queue[OutType]()
      val statePair: (Option[StateType], Option[StateType]) = (None, groupState.getOption)

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
          val out = nextState.map(s => stateToOutput(key, row, s)).toList.flatten
          (q.enqueue(out), (currentState, nextState))
        }
      }._1.toIterator
    }
  }

}