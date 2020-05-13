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

import scala.math.log


case class BernoulliDistribution(probability: Double) extends Distribution[Boolean, BernoulliDistribution] {

  override def loglikelihoods(samples: Seq[Boolean]): Seq[Double] = {
    samples.map(s => log(if (s) probability else 1 - probability))
  }

  override def scal(weight: Double): BernoulliDistribution = BernoulliDistribution(weight * probability)

  override def axpy(weight: Double, other: BernoulliDistribution): BernoulliDistribution = {
    BernoulliDistribution(weight * other.probability + probability)
  }

  override def summarize(weights: Seq[Double], samples: Seq[Boolean]): BernoulliDistribution = {
    val newRate = weights.zip(samples).foldLeft(0.0) {
      case(s, cur) => s + (if (cur._2) cur._1 else 0.0) / samples.length
    }
    BernoulliDistribution(newRate)
  }
}
