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


trait Distribution[SampleType, DistributionType <: Distribution[SampleType, DistributionType]] extends Product {

  def likelihood(sample: SampleType): Double

  def weightedDistribution(weight: Double): DistributionType

  def summarize(weights: Seq[Double], samples: Seq[SampleType], norm: Double): DistributionType

  def add(weight: Double, other: DistributionType): DistributionType
}

sealed trait MixtureDistribution[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  ImplType <: MixtureDistribution[SampleType, DistributionType, ImplType]] extends Product {

  def weights: Seq[Double]
  def distributions: Seq[DistributionType]

  def weightedDistributions: ImplType =  {
    val dists = distributions.zip(weights).map {
      case (dist, w) => dist.weightedDistribution(w)
    }
    asDistribution(weights, dists)
  }

  def reWeightedDistributions: ImplType = {
    val dists = distributions.zip(weights).map {
      case (dist, w) => dist.weightedDistribution(1.0/w)
    }
    asDistribution(weights, dists)
  }

  def weightedLikelihoods(samples: Seq[SampleType]): Seq[Seq[Double]] = {
    samples.map { sample =>
      val likelihoods = distributions.zip(weights).map {
        case (dist, weight) => dist.likelihood(sample) * weight
      }
      val sumLikelihood = likelihoods.sum
      likelihoods.map(_ / sumLikelihood)
    }.transpose
  }

  def weightedSummary(
    samples: Seq[SampleType], weight: Double): (Seq[Double], Seq[DistributionType]) = {
    val likelihoodWeights = weightedLikelihoods(samples)
    val sumLikelihoods = likelihoodWeights.flatten.sum

    val weightSummary= likelihoodWeights.map(s => weight * s.sum/s.length)
    val distsSummary = distributions.zip(likelihoodWeights).map {
      case (dist, w) => dist.summarize(w, samples, sumLikelihoods/weight)
    }

    (weightSummary, distsSummary)
  }

  def asDistribution(newWeights: Seq[Double], newDists: Seq[DistributionType]): ImplType = {
    this match {
      case t: GaussianMixtureDistribution => GaussianMixtureDistribution(
        newWeights, newDists.map(_.asInstanceOf[MultivariateGaussianDistribution])).asInstanceOf[ImplType]
      case t: PoissonMixtureDistribution => PoissonMixtureDistribution(
        newWeights, newDists.map(_.asInstanceOf[PoissonDistribution])).asInstanceOf[ImplType]
    }
  }

  def stochasticUpdate(
    mixture: MixtureDistribution[SampleType, DistributionType, ImplType],
    samples: Seq[SampleType],
    stepSize: Double): ImplType = {

    val (summaryWeights, summaryDists) = mixture.weightedSummary(samples, stepSize)
    val weightsSummary = weights
      .zip(summaryWeights).map(s => (1 - stepSize) * s._1 + s._2)

    val distsSummary = distributions.zip(summaryDists).map { case (left, right) =>
      right.add(1 - stepSize, left)
    }
    asDistribution(weightsSummary, distsSummary)
  }
}