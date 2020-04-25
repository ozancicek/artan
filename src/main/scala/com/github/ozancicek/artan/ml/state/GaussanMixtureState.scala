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

import org.apache.spark.ml.linalg.{Vector, DenseVector, DenseMatrix}
import java.sql.Timestamp

import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution


private[ml] case class GaussianMixtureInput(
    stateKey: String,
    measurement: Vector,
    stepSize: Double,
    initialMixtureModel: GaussianMixtureModel,
    eventTime: Option[Timestamp]) extends KeyedInput[String]

private[ml] case class GaussianMixtureState(
    stateIndex: Long,
    weightsSummary: Array[Double],
    meansSummary: Array[DenseVector],
    covariancesSummary: Array[DenseMatrix],
    mixtureModel: GaussianMixtureModel) extends State

case class GaussianMixtureModel(
    weights: Array[Double],
    distributions: Array[MultivariateGaussianDistribution])

case class GaussianMixtureOutput(
    stateKey: String,
    stateIndex: Long,
    mixtureModel: GaussianMixtureModel,
    eventTime: Option[Timestamp]) extends KeyedOutput[String]
