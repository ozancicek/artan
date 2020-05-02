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

import org.apache.spark.ml.BLAS
import org.apache.spark.ml.linalg.{DenseVector, Vector}


case class CategoricalDistribution(probabilities: Vector) extends Distribution[Int, CategoricalDistribution] {

  override def likelihood(sample: Int): Double = {
    val res = probabilities(sample)
    res
  }

  override def scal(weight: Double): CategoricalDistribution = {
    val newProbs = probabilities.copy
    BLAS.scal(weight, newProbs)
    CategoricalDistribution(newProbs)
  }

  override def axpy(weight: Double, other: CategoricalDistribution): CategoricalDistribution = {
    val newProbs = probabilities.copy
    BLAS.axpy(weight, other.probabilities, newProbs)
    CategoricalDistribution(newProbs)
  }

  override def summarize(weights: Seq[Double], samples: Seq[Int]): CategoricalDistribution = {
    val values = Array.fill(probabilities.size) {0.0}
    samples.zip(weights).foreach { case (v, d) =>
      values(v) = values(v) + d/samples.length
    }
    CategoricalDistribution(new DenseVector(values))
  }
}