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

import com.github.ozancicek.artan.ml.state.{MixtureInput, MixtureOutput, MixtureState, StateUpdateSpec}
import com.github.ozancicek.artan.ml.stats.{Distribution, MixtureDistribution}

private[mixture] trait MixtureParams[TransformerType]
  extends HasInitialWeights with HasInitialWeightsCol with HasStepSizeCol with HasStepSize with HasSampleCol
  with HasUpdateHoldout with HasMinibatchSize {

  def setInitialWeights(value: Array[Double]): TransformerType = set(initialWeights, value).asInstanceOf[TransformerType]

  def setInitialWeightsCol(value: String): TransformerType = set(initialWeightsCol, value).asInstanceOf[TransformerType]

  def setStepSize(value: Double): TransformerType = set(stepSize, value).asInstanceOf[TransformerType]

  def setStepSizeCol(value: String): TransformerType = set(stepSizeCol, value).asInstanceOf[TransformerType]

  def setSampleCol(value: String): TransformerType = set(sampleCol, value).asInstanceOf[TransformerType]

  def setUpdateHoldout(value: Int): TransformerType = set(updateHoldout, value).asInstanceOf[TransformerType]

  def setMinibatchSize(value: Int): TransformerType = set(minibatchSize, value).asInstanceOf[TransformerType]
}


private[mixture] trait MixtureUpdateSpec[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
  InputType <: MixtureInput[SampleType],
  StateType <: MixtureState[SampleType, DistributionType, MixtureType],
  OutputType <: MixtureOutput[SampleType, DistributionType, MixtureType]] extends StateUpdateSpec[
  String, InputType, StateType, OutputType]{

  def updateHoldout: Int

  def minibatchSize: Int

  protected def getInitialState(row: InputType): StateType

  protected def calculateNextState(
    row: InputType,
    state: Option[StateType]): (Long, List[SampleType], MixtureType, MixtureType) = {
    val currentState = state.getOrElse(getInitialState(row))
    val newSamples = row.sample :: currentState.samples

    val nextState = if (newSamples.size < minibatchSize) {
      (currentState.stateIndex, newSamples, currentState.summaryModel,currentState.mixtureModel)
    } else {
      val newSummaryModel = currentState
        .summaryModel
        .stochasticUpdate(currentState.mixtureModel, newSamples, row.stepSize)

      val newMixtureModel = if (currentState.stateIndex < updateHoldout) {
        currentState.mixtureModel
      } else {
        newSummaryModel.reWeightedDistributions
      }

      (currentState.stateIndex + 1, List.empty[SampleType], newSummaryModel, newMixtureModel)
    }
    nextState
  }
}