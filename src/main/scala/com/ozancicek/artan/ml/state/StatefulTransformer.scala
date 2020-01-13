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
import org.apache.spark.sql.streaming.{GroupState, OutputMode, GroupStateTimeout}
import org.apache.spark.sql._
import scala.collection.immutable.Queue
import scala.reflect.{ClassTag}
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
 */
private[ml] abstract class StatefulTransformer[
  GroupKeyType,
  RowType <: KeyedInput[GroupKeyType] : TypeTag,
  StateType <: KeyedState[GroupKeyType, OutType] : ClassTag,
  OutType <: Product : TypeTag] extends Transformer {

  /* Function to pass to flatMapGroupsWithState */
  protected def stateUpdateFunc: StateUpdateFunction[GroupKeyType, RowType, StateType, OutType]

  /* Keying function for groupByKey*/
  protected def keyFunc: RowType => GroupKeyType = (in: RowType) => in.stateKey

  /* State is encoded with kyro in order to support schema evolution. Others are encoded with spark encoders. */
  protected implicit val stateEncoder = Encoders.kryo[StateType]
  protected implicit val rowEncoder = Encoders.product[RowType]
  protected implicit val outEncoder = Encoders.product[OutType]

  protected def transformWithState(
    in: Dataset[RowType])(
    implicit keyEncoder: Encoder[GroupKeyType]): Dataset[OutType] = in
    .groupByKey(keyFunc)
    .flatMapGroupsWithState(
      OutputMode.Append,
      GroupStateTimeout.NoTimeout())(stateUpdateFunc)
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
  StateType <: KeyedState[GroupKeyType, OutType],
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
        (q :+ nextState.map(_.asOut), (currentState, nextState))
      }
    }._1.flatten.toIterator
  }
}