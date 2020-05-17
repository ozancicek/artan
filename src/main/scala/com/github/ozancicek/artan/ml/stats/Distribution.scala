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
import scala.math.{log, exp}

/**
 * Represents a multivariate gaussian mixture distribution
 *
 * @param weights Weight of each distribution, should sum up to 1.0
 * @param distributions Gaussian distributions
 */
case class GaussianMixtureDistribution(weights: Seq[Double], distributions: Seq[MultivariateGaussianDistribution])
  extends MixtureDistribution[Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution]

/**
 * Represents a poisson gaussian mixture distribution
 *
 * @param weights Weight of each distribution, should sum up to 1.0
 * @param distributions Poisson distributions
 */
case class PoissonMixtureDistribution(weights: Seq[Double], distributions: Seq[PoissonDistribution])
  extends MixtureDistribution[Long, PoissonDistribution, PoissonMixtureDistribution]

/**
 * Represents a bernoulli mixture distribution
 *
 * @param weights Weight of each distribution, should sum up to 1.0
 * @param distributions Bernoulli distributions
 */
case class BernoulliMixtureDistribution(weights: Seq[Double], distributions: Seq[BernoulliDistribution])
  extends MixtureDistribution[Boolean, BernoulliDistribution, BernoulliMixtureDistribution]

/**
 * Base trait for distribution
 *
 * @tparam SampleType Distribution sample type
 * @tparam DistributionType Distribution type
 */
private[artan] trait Distribution[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType]] extends Product {

  def loglikelihoods(samples: Seq[SampleType]): Seq[Double]

  /**
   * Scale the distribution parameter with a constant
   */
  private[artan] def scal(weight: Double): DistributionType

  /**
   * Returns sufficient statistics of weighted samples
   *
   * @param weights weight of the samples
   * @param samples sample sequence
   */
  private[artan] def summarize(weights: Seq[Double], samples: Seq[SampleType]): DistributionType

  /**
   * AXPY operation on distribution parameters
   */
  private[artan] def axpy(weight: Double, other: DistributionType): DistributionType
}


private[artan] sealed trait MixtureDistribution[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends Product {


  /**
   * Weights of the mixture distribution, should sum up to 1.0 and should align with [[distributions]].
   */
  def weights: Seq[Double]

  /**
   * Distributions of the mixture
   */
  def distributions: Seq[DistributionType]

  /**
   * Evaluates loglikelihood of the input sample sequence
   *
   * @param samples sample sequence
   * @return
   */
  def loglikelihood(samples: Seq[SampleType]): Double = {
    distributions.zip(weights).map { case (dist, weight) =>
      dist.loglikelihoods(samples).map(exp(_)*weight)
    }.transpose.map(s => log(s.sum)).sum
  }

  private def weightedLikelihoods(samples: Seq[SampleType]): Seq[Seq[Double]] = {

    val weightedLikelihoods = distributions.zip(weights).map { case (dist, weight) =>
      dist.loglikelihoods(samples).map(_ + log(weight))
    }

    val normedLikelihoods = weightedLikelihoods.transpose.map { distSamples =>
      // stable log sum for small probabilities
      // https://en.wikipedia.org/wiki/Log_probability#Addition_in_log_space
      val hd::tail = distSamples.sorted(Ordering.Double.reverse).toList

      val logProbSum = tail.foldLeft(hd) {
        case (sum, elem) => sum + log(1 + exp(elem - hd))
      }
      distSamples.map(ll => exp(ll - logProbSum))
    }
    normedLikelihoods.transpose
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
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]] extends Serializable {

  def create(weights: Seq[Double], dists: Seq[DistributionType]): MixtureType

}

private[artan] object MixtureDistribution {

  def factory[
    SampleType,
    DistributionType <: Distribution[SampleType, DistributionType],
    MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]](
    implicit mf: MixtureDistributionFactory[
      SampleType,
      DistributionType,
      MixtureType]): MixtureDistributionFactory[SampleType, DistributionType, MixtureType] = mf

  implicit val gaussianMD = new MixtureDistributionFactory[
    Vector, MultivariateGaussianDistribution, GaussianMixtureDistribution] {
    def create(weights: Seq[Double], dists: Seq[MultivariateGaussianDistribution]): GaussianMixtureDistribution = {
      GaussianMixtureDistribution(weights, dists)
    }
  }

  implicit val poissonMD = new MixtureDistributionFactory[
    Long, PoissonDistribution, PoissonMixtureDistribution] {
    def create(weights: Seq[Double], dists: Seq[PoissonDistribution]): PoissonMixtureDistribution = {
      PoissonMixtureDistribution(weights, dists)
    }
  }

  implicit val bernoulliMD = new MixtureDistributionFactory[
    Boolean, BernoulliDistribution, BernoulliMixtureDistribution] {
    def create(weights: Seq[Double], dists: Seq[BernoulliDistribution]): BernoulliMixtureDistribution = {
      BernoulliMixtureDistribution(weights, dists)
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