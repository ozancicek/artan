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


sealed trait MixtureState[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends State {

  def samples: List[SampleType]

  def mixtureModel: MixtureType

  def summaryModel: MixtureType
}


sealed trait MixtureInput[SampleType] extends KeyedInput[String] {

  def sample: SampleType

  def stepSize: Double
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
  extends MixtureInput[Vector]


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


private[ml] case class PoissonMixtureInput(
    stateKey: String,
    sample: Long,
    stepSize: Double,
    initialMixtureModel: PoissonMixtureDistribution,
    eventTime: Option[Timestamp])
  extends MixtureInput[Long]


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