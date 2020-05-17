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

import org.apache.spark.ml.linalg.Vector
import java.sql.Timestamp

import com.github.ozancicek.artan.ml.stats._


private[ml] sealed trait MixtureState[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends State {

  def samples: List[SampleType]

  def mixtureModel: MixtureType

  def summaryModel: MixtureType

  def loglikelihood: Double

}


private[ml] sealed trait MixtureInput[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends KeyedInput[String] {

  def sample: SampleType

  def stepSize: Double

  def decayRate: Option[Double]

  def minibatchSize: Long

  def updateHoldout: Long

  def initialMixtureModel: MixtureType
}


private[ml] sealed trait MixtureOutput[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends KeyedOutput[String] {

  def mixtureModel: MixtureType

  def loglikelihood: Double
}


private[ml] case class GaussianMixtureInput(
    stateKey: String,
    sample: Vector,
    stepSize: Double,
    decayRate: Option[Double],
    minibatchSize: Long,
    updateHoldout: Long,
    initialMixtureModel: GaussianMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]


private[ml] case class GaussianMixtureState(
    stateIndex: Long,
    samples: List[Vector],
    summaryModel: GaussianMixtureDistribution,
    mixtureModel: GaussianMixtureDistribution,
    loglikelihood: Double)
  extends MixtureState[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]

/**
 * Case class representing the output of a gaussian mixture estimation
 *
 * @param stateKey Key of the state
 * @param stateIndex index of the filter, incremented on each minibatch of samples
 * @param mixtureModel mixture model estimation with weights and distributions
 * @param eventTime event time of the input measurement
 * @param loglikelihood likelihood of samples processed at this state (not cumulative)
 */
case class GaussianMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: GaussianMixtureDistribution,
    eventTime: Option[Timestamp],
    loglikelihood: Double)
  extends MixtureOutput[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]


private[ml] case class BernoulliMixtureInput(
    stateKey: String,
    sample: Boolean,
    stepSize: Double,
    decayRate: Option[Double],
    minibatchSize: Long,
    updateHoldout: Long,
    initialMixtureModel: BernoulliMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Boolean, BernoulliDistribution, BernoulliMixtureDistribution]


private[ml] case class BernoulliMixtureState(
    stateIndex: Long,
    samples: List[Boolean],
    summaryModel: BernoulliMixtureDistribution,
    mixtureModel: BernoulliMixtureDistribution,
    loglikelihood: Double)
  extends MixtureState[Boolean, BernoulliDistribution, BernoulliMixtureDistribution]

/**
 * Case class representing the output of a bernoulli mixture estimation
 *
 * @param stateKey Key of the state
 * @param stateIndex index of the filter, incremented on each minibatch of samples
 * @param mixtureModel mixture model estimation with weights and distributions
 * @param eventTime event time of the input measurement
 * @param loglikelihood likelihood of samples processed at this state (not cumulative)
 */
case class BernoulliMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: BernoulliMixtureDistribution,
    eventTime: Option[Timestamp],
    loglikelihood: Double)
  extends MixtureOutput[Boolean, BernoulliDistribution, BernoulliMixtureDistribution]


private[ml] case class PoissonMixtureInput(
    stateKey: String,
    sample: Long,
    stepSize: Double,
    decayRate: Option[Double],
    minibatchSize: Long,
    updateHoldout: Long,
    initialMixtureModel: PoissonMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Long, PoissonDistribution, PoissonMixtureDistribution]


private[ml] case class PoissonMixtureState(
    stateIndex: Long,
    samples: List[Long],
    summaryModel: PoissonMixtureDistribution,
    mixtureModel: PoissonMixtureDistribution,
    loglikelihood: Double)
  extends MixtureState[Long, PoissonDistribution, PoissonMixtureDistribution]

/**
 * Case class representing the output of a poisson mixture estimation
 *
 * @param stateKey Key of the state
 * @param stateIndex index of the filter, incremented on each minibatch of samples
 * @param mixtureModel mixture model estimation with weights and distributions
 * @param eventTime event time of the input measurement
 * @param loglikelihood likelihood of samples processed at this state (not cumulative)
 */
case class PoissonMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: PoissonMixtureDistribution,
    eventTime: Option[Timestamp],
    loglikelihood: Double)
  extends MixtureOutput[Long, PoissonDistribution, PoissonMixtureDistribution]


private[ml] trait MixtureStateFactory[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
  StateType <: MixtureState[SampleType, DistributionType, MixtureType],
  OutType <: MixtureOutput[SampleType, DistributionType, MixtureType]] extends Serializable {

  def createState(
    stateIndex: Long,
    samples: List[SampleType],
    summary: MixtureType,
    mixture: MixtureType,
    loglikelihood: Double): StateType

  def createOutput(
    stateKey: String,
    stateIndex: Long,
    mixture: MixtureType,
    eventTime: Option[Timestamp],
    loglikelihood: Double): OutType

}


/**
 * Helper object to generically create mixture states.
 */
private[ml] object MixtureStateFactory {

  def apply[
    SampleType,
    DistributionType <: Distribution[SampleType, DistributionType],
    MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
    StateType <: MixtureState[SampleType, DistributionType, MixtureType],
    OutType <: MixtureOutput[SampleType, DistributionType, MixtureType]](implicit msf: MixtureStateFactory[
    SampleType, DistributionType, MixtureType, StateType, OutType]): MixtureStateFactory[
    SampleType, DistributionType, MixtureType, StateType, OutType] = msf


  implicit val gaussianSF: MixtureStateFactory[
    Vector, MultivariateGaussianDistribution,
    GaussianMixtureDistribution, GaussianMixtureState, GaussianMixtureOutput] = new MixtureStateFactory[
    Vector, MultivariateGaussianDistribution,
    GaussianMixtureDistribution, GaussianMixtureState, GaussianMixtureOutput] {

    def createState(
      stateIndex: Long,
      samples: List[Vector],
      summary: GaussianMixtureDistribution,
      mixture: GaussianMixtureDistribution,
      loglikelihood: Double): GaussianMixtureState = {
      GaussianMixtureState(stateIndex, samples, summary, mixture, loglikelihood)
    }

    override def createOutput(
      stateKey: String,
      stateIndex: Long,
      mixture: GaussianMixtureDistribution,
      eventTime: Option[Timestamp],
      loglikelihood: Double): GaussianMixtureOutput =  {
      GaussianMixtureOutput(stateKey, stateIndex, mixture, eventTime, loglikelihood)
    }

  }

  implicit val poissonSF: MixtureStateFactory[
    Long, PoissonDistribution,
    PoissonMixtureDistribution, PoissonMixtureState, PoissonMixtureOutput] = new MixtureStateFactory[
    Long, PoissonDistribution,
    PoissonMixtureDistribution, PoissonMixtureState, PoissonMixtureOutput] {

    override def createOutput(
      stateKey: String,
      stateIndex: Long,
      mixture: PoissonMixtureDistribution,
      eventTime: Option[Timestamp],
      loglikelihood: Double): PoissonMixtureOutput = {
      PoissonMixtureOutput(stateKey, stateIndex, mixture, eventTime, loglikelihood)
    }

    def createState(
      stateIndex: Long,
      samples: List[Long],
      summary: PoissonMixtureDistribution,
      mixture: PoissonMixtureDistribution,
      loglikelihood: Double): PoissonMixtureState = {
      PoissonMixtureState(stateIndex, samples, summary, mixture, loglikelihood)
    }
  }

  implicit val bernoulliSF: MixtureStateFactory[
    Boolean, BernoulliDistribution,
    BernoulliMixtureDistribution, BernoulliMixtureState, BernoulliMixtureOutput] = new MixtureStateFactory[
    Boolean, BernoulliDistribution,
    BernoulliMixtureDistribution, BernoulliMixtureState, BernoulliMixtureOutput] {

    override def createOutput(
      stateKey: String,
      stateIndex: Long,
      mixture: BernoulliMixtureDistribution,
      eventTime: Option[Timestamp],
      loglikelihood: Double): BernoulliMixtureOutput = {
      BernoulliMixtureOutput(stateKey, stateIndex, mixture, eventTime, loglikelihood)
    }

    def createState(
      stateIndex: Long,
      samples: List[Boolean],
      summary: BernoulliMixtureDistribution,
      mixture: BernoulliMixtureDistribution,
      loglikelihood: Double): BernoulliMixtureState = {
      BernoulliMixtureState(stateIndex, samples, summary, mixture, loglikelihood)
    }
  }

}
