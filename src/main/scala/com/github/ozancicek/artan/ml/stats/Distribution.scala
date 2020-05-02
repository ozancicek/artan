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


package com.github.ozancicek.artan.ml.stats

import org.apache.spark.ml.linalg.Vector


case class GaussianMixtureDistribution(weights: Seq[Double], distributions: Seq[MultivariateGaussianDistribution])
  extends MixtureDistribution[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]


case class PoissonMixtureDistribution(weights: Seq[Double], distributions: Seq[PoissonDistribution])
  extends MixtureDistribution[Long, PoissonDistribution, PoissonMixtureDistribution]


case class CategoricalMixtureDistribution(weights: Seq[Double], distributions: Seq[CategoricalDistribution])
  extends MixtureDistribution[Int, CategoricalDistribution, CategoricalMixtureDistribution]


trait Distribution[SampleType, DistributionType <: Distribution[SampleType, DistributionType]] extends Product {

  def likelihood(sample: SampleType): Double

  def scal(weight: Double): DistributionType

  def summarize(weights: Seq[Double], samples: Seq[SampleType]): DistributionType

  def axpy(weight: Double, other: DistributionType): DistributionType
}


sealed trait MixtureDistribution[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends Product {

  def weights: Seq[Double]
  def distributions: Seq[DistributionType]

  def weightedDistributions(
    implicit mdf: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]): MixtureType =  {
    val dists = distributions.zip(weights).map {
      case (dist, w) => dist.scal(w)
    }
    mdf.create(weights, dists)
  }

  def reWeightedDistributions(
    implicit mdf: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]): MixtureType = {
    val dists = distributions.zip(weights).map {
      case (dist, w) => dist.scal(1.0/w)
    }
    mdf.create(weights, dists)
  }

  private def weightedLikelihoods(samples: Seq[SampleType]): Seq[Seq[Double]] = {
    samples.map { sample =>
      val likelihoods = distributions.zip(weights).map {
        case (dist, weight) => dist.likelihood(sample) * weight
      }
      val sumLikelihood = likelihoods.sum
      likelihoods.map(_ / sumLikelihood)
    }.transpose
  }

  private def summary(
    samples: Seq[SampleType])(
    implicit mdf: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]): MixtureType = {
    val likelihoodWeights = weightedLikelihoods(samples)

    val weightSummary = likelihoodWeights.map(s => s.sum/samples.length)
    val distsSummary = distributions.zip(likelihoodWeights).map {
      case (dist, w) => dist.summarize(w, samples)
    }

    mdf.create(weightSummary, distsSummary)
  }
}


private[artan] trait MixtureDistributionFactory[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]]{

  def create(weights: Seq[Double], dists: Seq[DistributionType]): MixtureType

}

object MixtureDistribution {

  def factory[
    SampleType,
    DistributionType <: Distribution[SampleType, DistributionType],
    MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]](
    implicit mf: MixtureDistributionFactory[
      SampleType,
      DistributionType,
      MixtureType]): MixtureDistributionFactory[SampleType, DistributionType, MixtureType] = mf

  implicit val gaussianMD: MixtureDistributionFactory[
    Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution] = new MixtureDistributionFactory[
    Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution] {
    def create(weights: Seq[Double], dists: Seq[MultivariateGaussianDistribution]): GaussianMixtureDistribution = {
      GaussianMixtureDistribution(weights, dists)
    }
  }

  implicit val poissonMD: MixtureDistributionFactory[
    Long, PoissonDistribution, PoissonMixtureDistribution] = new MixtureDistributionFactory[
    Long, PoissonDistribution, PoissonMixtureDistribution] {
    def create(weights: Seq[Double], dists: Seq[PoissonDistribution]): PoissonMixtureDistribution = {
      PoissonMixtureDistribution(weights, dists)
    }
  }

  implicit val categoricalMD: MixtureDistributionFactory[
    Int, CategoricalDistribution, CategoricalMixtureDistribution] = new MixtureDistributionFactory[
    Int, CategoricalDistribution, CategoricalMixtureDistribution] {
    def create(weights: Seq[Double], dists: Seq[CategoricalDistribution]): CategoricalMixtureDistribution = {
      CategoricalMixtureDistribution(weights, dists)
    }
  }

  def stochasticUpdate[
    SampleType,
    DistributionType <: Distribution[SampleType, DistributionType],
    MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]](
    summary: MixtureType, mixture: MixtureType, samples: Seq[SampleType], stepSize: Double)(
    implicit mf: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]): MixtureType = {

    val summaryDist = mixture.summary(samples)

    val weightsSummary = summary.weights
      .zip(summaryDist.weights).map(s => (1 - stepSize) * s._1 + stepSize * s._2)
    val distsSummary = summary.distributions.zip(summaryDist.distributions).map { case (left, right) =>
      right.scal(stepSize).axpy(1 - stepSize, left)
    }

    mf.create(weightsSummary, distsSummary)
  }

  def weightedMixture[
    SampleType,
    DistributionType <: Distribution[SampleType, DistributionType],
    MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]](
    mixture: MixtureType)(
    implicit mdf: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]): MixtureType =  {
    val dists = mixture.distributions.zip(mixture.weights).map {
      case (dist, w) => dist.scal(w)
    }
    mdf.create(mixture.weights, dists)
  }

  def inverseWeightedMixture[
    SampleType,
    DistributionType <: Distribution[SampleType, DistributionType],
    MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]](
    mixture: MixtureType)(
    implicit mdf: MixtureDistributionFactory[SampleType, DistributionType, MixtureType]): MixtureType =  {
    val dists = mixture.distributions.zip(mixture.weights).map {
      case (dist, w) => dist.scal(1.0/w)
    }
    mdf.create(mixture.weights, dists)
  }
}