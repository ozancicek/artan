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

package com.github.ozancicek.artan.ml.em

import org.apache.spark.ml.param._


private[em] trait HasInitialWeights extends Params {

  def mixtureCount: Int

  final val initialWeights: Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialWeights",
    "initialWeights")

  setDefault(initialWeights, Array.fill(mixtureCount) {1.0/mixtureCount})

  final def getInitialWeights: Array[Double] = $(initialWeights)
}


private[em] trait HasInitialWeightsCol extends Params {

  final val initialWeightsCol: Param[String] = new Param[String](
    this,
    "initialWeightsCol",
    "initialWeightsCol"
  )

  final def getInitialWeightsCol: String = $(initialWeightsCol)
}

private[em] trait HasStepSize extends Params {

  final val stepSize: Param[Double] = new DoubleParam(
    this,
    "stepSize",
    "stepSize",
    ParamValidators.lt(1.0)
  )

  setDefault(stepSize, 0.01)

  final def getStepSize: Double = $(stepSize)
}


private[em] trait HasStepSizeCol extends Params {

  final val stepSizeCol: Param[String] = new Param[String](
    this,
    "stepSizeCol",
    "stepSizeCol")

  final def getStepSizeCol: String = $(stepSizeCol)
}