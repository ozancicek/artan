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

import com.google.common.math.BigIntegerMath
import scala.math.log

/**
 * Represents a poisson distribution with a single parameter
 *
 * @param rate Rate parameter.
 */
case class PoissonDistribution(rate: Double) extends Distribution[Long, PoissonDistribution] {

  /**
   * Returns logpmf of sample sequence
   *
   * @param samples poisson sample sequence
   */
  def loglikelihoods(samples: Seq[Long]): Seq[Double] = samples.map(i => Poisson.logpmf(i, rate))

  private[artan] def scal(weight: Double): PoissonDistribution = PoissonDistribution(weight * rate)

  private[artan] def axpy(weight: Double, other: PoissonDistribution): PoissonDistribution = {
    PoissonDistribution(other.rate * weight + rate)
  }

  private[artan] def summarize(weights: Seq[Double], samples: Seq[Long]): PoissonDistribution = {
    val newRate = weights.zip(samples).foldLeft(0.0) {
      case(s, cur) => s + cur._1 * cur._2 /samples.length
    }
    PoissonDistribution(newRate)
  }

}

private[artan] object Poisson {

  def logpmf(count: Long, rate:Double): Double = {
    val fac = BigIntegerMath.factorial(count.toInt).doubleValue()
    -rate -log(fac) + log(rate)*count
  }

}
