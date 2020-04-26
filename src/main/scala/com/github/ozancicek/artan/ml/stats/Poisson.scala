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
import scala.math.{exp, pow}


case class PoissonDistribution(rate: Double) extends Distribution[Long, PoissonDistribution] {

  override def likelihood(sample: Long): Double = pmf(sample)

  override def weightedDistribution(weight: Double): PoissonDistribution = PoissonDistribution(weight * rate)

  override def add(weight: Double, other: PoissonDistribution): PoissonDistribution = {
    PoissonDistribution(other.rate * weight + rate)
  }

  override def summarize(weights: Seq[Double], samples: Seq[Long], norm: Double): PoissonDistribution = {
    val newRate = weights.zip(samples).foldLeft(0.0) {
      case(s, cur) => s + cur._1 * cur._2 / norm
    }
    PoissonDistribution(newRate)
  }

  def pmf(count: Long): Double = Poisson.pmf(count, rate)
}

private[ml] object Poisson {

  def pmf(count: Long, rate: Double): Double = {
    val fac = BigIntegerMath.factorial(count.toInt).doubleValue()
    pow(rate, count) * exp(-rate) / fac
  }

}
