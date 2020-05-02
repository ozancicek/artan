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
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization


sealed trait MixtureState[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends State {

  def samples: List[SampleType]

  def mixtureModel: MixtureType

  def summaryModel: MixtureType

}


sealed trait MixtureInput[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends KeyedInput[String] {

  def sample: SampleType

  def stepSize: Double

  def initialMixtureModel: MixtureType
}


sealed trait MixtureOutput[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends KeyedOutput[String] {

  def mixtureModel: MixtureType
}


private[ml] case class GaussianMixtureInput(
    stateKey: String,
    sample: Vector,
    stepSize: Double,
    initialMixtureModel: GaussianMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]


private[ml] case class GaussianMixtureState(
    stateIndex: Long,
    samples: List[Vector],
    summaryModel: GaussianMixtureDistribution,
    mixtureModel: GaussianMixtureDistribution)
  extends MixtureState[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]


case class GaussianMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: GaussianMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureOutput[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]


private[ml] case class CategoricalMixtureInput(
    stateKey: String,
    sample: Int,
    stepSize: Double,
    initialMixtureModel: CategoricalMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Int, CategoricalDistribution, CategoricalMixtureDistribution]


private[ml] case class CategoricalMixtureState(
    stateIndex: Long,
    samples: List[Int],
    summaryModel: CategoricalMixtureDistribution,
    mixtureModel: CategoricalMixtureDistribution)
  extends MixtureState[Int, CategoricalDistribution, CategoricalMixtureDistribution]


case class CategoricalMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: CategoricalMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureOutput[Int, CategoricalDistribution, CategoricalMixtureDistribution]


private[ml] case class PoissonMixtureInput(
    stateKey: String,
    sample: Long,
    stepSize: Double,
    initialMixtureModel: PoissonMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Long, PoissonDistribution, PoissonMixtureDistribution]


private[ml] case class PoissonMixtureState(
    stateIndex: Long,
    samples: List[Long],
    summaryModel: PoissonMixtureDistribution,
    mixtureModel: PoissonMixtureDistribution)
  extends MixtureState[Long, PoissonDistribution, PoissonMixtureDistribution]


case class PoissonMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: PoissonMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureOutput[Long, PoissonDistribution, PoissonMixtureDistribution]


private[artan] trait MixtureStateFactory[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
  StateType <: MixtureState[SampleType, DistributionType, MixtureType],
  OutType <: MixtureOutput[SampleType, DistributionType, MixtureType]] extends Serializable {

  implicit def distributionFactory: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]

  def createState(
    stateIndex: Long, samples: List[SampleType], summary: MixtureType, mixture: MixtureType): StateType

  def createOutput(
    stateKey: String, stateIndex: Long, mixture: MixtureType, eventTime: Option[Timestamp]): OutType

}


object MixtureStateFactory {

  implicit val gaussianSF: MixtureStateFactory[
    Vector, MultivariateGaussianDistribution,
    GaussianMixtureDistribution, GaussianMixtureState, GaussianMixtureOutput] = new MixtureStateFactory[
    Vector, MultivariateGaussianDistribution,
    GaussianMixtureDistribution, GaussianMixtureState, GaussianMixtureOutput] {

    override implicit def distributionFactory: MixtureDistributionFactory[
      Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution] = MixtureDistribution.gaussianMD

    def createState(
      stateIndex: Long,
      samples: List[Vector],
      summary: GaussianMixtureDistribution,
      mixture: GaussianMixtureDistribution): GaussianMixtureState = {
      GaussianMixtureState(stateIndex, samples, summary, mixture)
    }

    override def createOutput(
      stateKey: String,
      stateIndex: Long,
      mixture: GaussianMixtureDistribution,
      eventTime: Option[Timestamp]): GaussianMixtureOutput =  {
      GaussianMixtureOutput(stateKey, stateIndex, mixture, eventTime)
    }

  }

  implicit val poissonSF: MixtureStateFactory[
    Long, PoissonDistribution,
    PoissonMixtureDistribution, PoissonMixtureState, PoissonMixtureOutput] = new MixtureStateFactory[
    Long, PoissonDistribution,
    PoissonMixtureDistribution, PoissonMixtureState, PoissonMixtureOutput] {

    override implicit def distributionFactory: MixtureDistributionFactory[
      Long, PoissonDistribution, PoissonMixtureDistribution] = MixtureDistribution.poissonMD

    override def createOutput(
      stateKey: String,
      stateIndex: Long,
      mixture: PoissonMixtureDistribution,
      eventTime: Option[Timestamp]): PoissonMixtureOutput = {
      PoissonMixtureOutput(stateKey, stateIndex, mixture, eventTime)
    }

    def createState(
      stateIndex: Long,
      samples: List[Long],
      summary: PoissonMixtureDistribution,
      mixture: PoissonMixtureDistribution): PoissonMixtureState = {
      PoissonMixtureState(stateIndex, samples, summary, mixture)
    }
  }

  implicit val categoricalSF: MixtureStateFactory[
    Int, CategoricalDistribution,
    CategoricalMixtureDistribution, CategoricalMixtureState, CategoricalMixtureOutput] = new MixtureStateFactory[
    Int, CategoricalDistribution,
    CategoricalMixtureDistribution, CategoricalMixtureState, CategoricalMixtureOutput] {

    override implicit def distributionFactory: MixtureDistributionFactory[
      Int, CategoricalDistribution, CategoricalMixtureDistribution] = MixtureDistribution.categoricalMD

    override def createOutput(
      stateKey: String,
      stateIndex: Long,
      mixture: CategoricalMixtureDistribution,
      eventTime: Option[Timestamp]): CategoricalMixtureOutput = {
      CategoricalMixtureOutput(stateKey, stateIndex, mixture, eventTime)
    }
    def createState(
      stateIndex: Long,
      samples: List[Int],
      summary: CategoricalMixtureDistribution,
      mixture: CategoricalMixtureDistribution): CategoricalMixtureState = {
      CategoricalMixtureState(stateIndex, samples, summary, mixture)
    }
  }

}