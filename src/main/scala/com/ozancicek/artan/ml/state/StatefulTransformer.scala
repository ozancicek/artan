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
