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


private[ml] trait StateUpdateFunction[
  GroupKeyType,
  RowType,
  StateType]
  extends Function3[
  GroupKeyType,
  Iterator[RowType],
  GroupState[StateType],
  Iterator[StateType]] with Serializable {

  def updateGroupState(
    key: GroupKeyType,
    row: RowType,
    state: Option[StateType]): Option[StateType]

  def apply(
    key: GroupKeyType,
    rows: Iterator[RowType],
    groupState: GroupState[StateType]
  ): Iterator[StateType] = {

  if (groupState.hasTimedOut) {
    groupState.remove()
  }

  val currentState = groupState.getOption
  val nextState = rows.foldLeft(List(currentState)) {
    case (states, row) => updateGroupState(key, row, states.head) :: states
  }
  nextState.head.foreach(s=>groupState.update(s))
  nextState.flatten.reverse.toIterator
  }
}


private[ml] trait StatefulTransformer[
  GroupKeyType,
  RowType,
  StateType] extends Transformer {

  def stateUpdateFunc: StateUpdateFunction[GroupKeyType, RowType, StateType]
  def keyFunc: (RowType) => GroupKeyType

  def transformWithState(
    in: Dataset[RowType])(implicit
    keyEncoder: Encoder[GroupKeyType],
    rowEncoder: Encoder[RowType],
    stateEncoder: Encoder[StateType]) = in
      .groupByKey(keyFunc)
      .flatMapGroupsWithState(
        OutputMode.Append,
        GroupStateTimeout.ProcessingTimeTimeout())(stateUpdateFunc)
}
