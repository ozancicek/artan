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

/**
 * Base trait for common mixture parameters
 *
 * @tparam TransformerType Type of the mixture transformer
 */
private[mixture] trait MixtureParams[TransformerType]
  extends HasInitialWeights with HasInitialWeightsCol with HasStepSizeCol with HasStepSize with HasSampleCol
  with HasUpdateHoldout with HasMinibatchSize with HasDecayRate {

  /**
   * Sets the initial weights of the mixtures. The weights should sum up to 1.0.
   *
   * @group setParam
   */
  def setInitialWeights(value: Array[Double]): TransformerType = set(initialWeights, value)
    .asInstanceOf[TransformerType]

  /**
   * Sets the initial weights of the mixtures from dataframe column. Column should contain array of doubles.
   * Overrides the value set by [[setInitialWeights]].
   *
   * @group setParam
   */
  def setInitialWeightsCol(value: String): TransformerType = set(initialWeightsCol, value)
    .asInstanceOf[TransformerType]

  /**
   * Sets the step size parameter, which weights the current parameter of the model against the old parameter.
   * A step size of 1.0 means ignore the old parameter, whereas a step size of 0 means ignore the current parameter.
   * Values closer to 1.0 will increase speed of convergence, but might have adverse effects on stability. In an online
   * setting, its advised to set it close to 0.0.
   *
   * Default is 0.1
   *
   * @group setParam
   */
  def setStepSize(value: Double): TransformerType = set(stepSize, value).asInstanceOf[TransformerType]

  /**
   * Sets the step size from dataframe column, which would allow setting different step sizes accross measurements.
   * Overrides the value set by [[setStepSize]].
   *
   * @group setParam
   */
  def setStepSizeCol(value: String): TransformerType = set(stepSizeCol, value).asInstanceOf[TransformerType]

  /**
   * Sets the sample column for the mixture model inputs. Depending on the mixture distribution, sample type should
   * be different.
   *
   * Bernoulli => Boolean
   * Poisson => Long
   * MultivariateGaussian => Vector
   *
   * @group setParam
   */
  def setSampleCol(value: String): TransformerType = set(sampleCol, value).asInstanceOf[TransformerType]

  /**
   * Sets the update holdout parameter which controls after how many samples the mixture will start calculating
   * estimates. Preventing update in first few samples might be preferred for stability.
   *
   * @group setParam
   */
  def setUpdateHoldout(value: Int): TransformerType = set(updateHoldout, value).asInstanceOf[TransformerType]

  /**
   * Sets the minibatch size for batching samples together in online EM algorithm. Estimate will be produced once
   * per each batch. Having larger batches increases stability with increased memory footprint.
   *
   * Default is 1
   *
   * @group setParam
   */
  def setMinibatchSize(value: Int): TransformerType = set(minibatchSize, value).asInstanceOf[TransformerType]

  /**
   * Sets the step size as a decaying function rather than a constant step size, which might be preferred
   * for batch training. If set, the step size will be replaced with the output of following function:
   *
   * stepSize = pow(2 + kIter, -decayRate)
   *
   * Where kIter is incremented by 1 at each minibatch.
   *
   * @group setParam
   */
  def setDecayRate(value: Double): TransformerType = set(decayRate, value).asInstanceOf[TransformerType]

}

/**
 * State update spec of online EM for finite mixtures.
 */
private[mixture] class MixtureUpdateSpec[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
  InputType <: MixtureInput[SampleType, DistributionType, MixtureType],
  StateType <: MixtureState[SampleType, DistributionType, MixtureType],
  OutputType <: MixtureOutput[SampleType, DistributionType, MixtureType]](
  updateHoldout: Int, minibatchSize: Int)(implicit
  stateFactory: MixtureStateFactory[
    SampleType, DistributionType, MixtureType, StateType, OutputType],
  mixtureFactory: MixtureDistributionFactory[
    SampleType, DistributionType, MixtureType]) extends StateUpdateSpec[String, InputType, StateType, OutputType]{

  protected def stateToOutput(
    key: String,
    row: InputType,
    state: StateType): List[OutputType] = {
    // Output once per minibatch, which is tracked from #samples stored in state
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

      val stepSize = row.decayRate match {
        case Some(rate) => pow(2 + currentState.stateIndex, -rate)
        case None => row.stepSize
      }

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