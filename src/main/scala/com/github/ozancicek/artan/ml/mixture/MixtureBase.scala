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

package com.github.ozancicek.artan.ml.mixture

import com.github.ozancicek.artan.ml.state.{MixtureInput, MixtureOutput, MixtureState, MixtureStateFactory, StateUpdateSpec}
import com.github.ozancicek.artan.ml.stats.{Distribution, MixtureDistribution, MixtureDistributionFactory}

import scala.math.pow


private[mixture] trait MixtureParams[TransformerType]
  extends HasInitialWeights with HasInitialWeightsCol with HasStepSizeCol with HasStepSize with HasSampleCol
  with HasUpdateHoldout with HasMinibatchSize with HasDecayingStepSizeEnabled {

  def setInitialWeights(value: Array[Double]): TransformerType = set(initialWeights, value)
    .asInstanceOf[TransformerType]

  def setInitialWeightsCol(value: String): TransformerType = set(initialWeightsCol, value)
    .asInstanceOf[TransformerType]

  def setStepSize(value: Double): TransformerType = set(stepSize, value).asInstanceOf[TransformerType]

  def setStepSizeCol(value: String): TransformerType = set(stepSizeCol, value).asInstanceOf[TransformerType]

  def setSampleCol(value: String): TransformerType = set(sampleCol, value).asInstanceOf[TransformerType]

  def setUpdateHoldout(value: Int): TransformerType = set(updateHoldout, value).asInstanceOf[TransformerType]

  def setMinibatchSize(value: Int): TransformerType = set(minibatchSize, value).asInstanceOf[TransformerType]

  def setEnableDecayingStepSize: TransformerType = set(decayingStepSizeEnabled, true).asInstanceOf[TransformerType]

}


private[mixture] class MixtureUpdateSpec[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
  InputType <: MixtureInput[SampleType, DistributionType, MixtureType],
  StateType <: MixtureState[SampleType, DistributionType, MixtureType],
  OutputType <: MixtureOutput[SampleType, DistributionType, MixtureType]](
  updateHoldout: Int, minibatchSize: Int, decayingStepSize: Boolean)(implicit
  stateFactory: MixtureStateFactory[
    SampleType, DistributionType, MixtureType, StateType, OutputType],
  mixtureFactory: MixtureDistributionFactory[
    SampleType, DistributionType, MixtureType]) extends StateUpdateSpec[String, InputType, StateType, OutputType]{

  protected def stateToOutput(
    key: String,
    row: InputType,
    state: StateType): List[OutputType] = {
    if (state.samples.isEmpty) {
      List(stateFactory.createOutput(
        key,
        state.stateIndex,
        state.mixtureModel,
        row.eventTime))
    } else {
      List.empty[OutputType]
    }
  }

  protected def getInitialState(row: InputType): StateType = {
    val weightedDist = MixtureDistribution.weightedMixture[SampleType, DistributionType, MixtureType](
      row.initialMixtureModel)
    stateFactory.createState(
      0L,
      List.empty[SampleType],
      weightedDist,
      row.initialMixtureModel
    )
  }

  protected def calculateNextState(
    row: InputType,
    state: Option[StateType]): Option[StateType] = {
    val currentState = state.getOrElse(getInitialState(row))
    val newSamples = row.sample :: currentState.samples

    val nextState = if (newSamples.size < minibatchSize) {
      stateFactory.createState(
        currentState.stateIndex, newSamples, currentState.summaryModel, currentState.mixtureModel)
    } else {

      val stepSize = if (decayingStepSize) pow(2 + currentState.stateIndex, -row.stepSize) else row.stepSize

      val newSummaryModel = MixtureDistribution
        .stochasticUpdate[SampleType, DistributionType, MixtureType](
        currentState.summaryModel, currentState.mixtureModel, newSamples, stepSize)

      val newMixtureModel = if (currentState.stateIndex < updateHoldout) {
        currentState.mixtureModel
      } else {
        MixtureDistribution.inverseWeightedMixture[SampleType, DistributionType, MixtureType](newSummaryModel)
      }

      stateFactory.createState(currentState.stateIndex + 1, List.empty[SampleType], newSummaryModel, newMixtureModel)
    }
    Some(nextState)
  }

  protected def updateGroupState(
    key: String,
    row: InputType,
    state: Option[StateType]): Option[StateType] = calculateNextState(row, state)


}