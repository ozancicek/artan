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

package com.github.ozancicek.artan.ml.mixture

import org.apache.spark.ml.param._


private[mixture] trait HasInitialWeights extends Params {

  def mixtureCount: Int

  final val initialWeights: Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialWeights",
    "initialWeights")

  setDefault(initialWeights, Array.fill(mixtureCount) {1.0/mixtureCount})

  final def getInitialWeights: Array[Double] = $(initialWeights)
}


private[mixture] trait HasInitialWeightsCol extends Params {

  final val initialWeightsCol: Param[String] = new Param[String](
    this,
    "initialWeightsCol",
    "initialWeightsCol"
  )

  final def getInitialWeightsCol: String = $(initialWeightsCol)
}

private[mixture] trait HasStepSize extends Params {

  final val stepSize: Param[Double] = new DoubleParam(
    this,
    "stepSize",
    "stepSize",
    ParamValidators.lt(1.0)
  )

  setDefault(stepSize, 0.01)

  final def getStepSize: Double = $(stepSize)
}


private[mixture] trait HasStepSizeCol extends Params {

  final val stepSizeCol: Param[String] = new Param[String](
    this,
    "stepSizeCol",
    "stepSizeCol")

  final def getStepSizeCol: String = $(stepSizeCol)
}


private[mixture] trait HasDecayingStepSizeEnabled extends Params {

  final val decayingStepSizeEnabled: Param[Boolean] = new BooleanParam(
    this,
    "decayingStepSizeEnabled",
    "decayingStepSizeEnabled"
  )
  setDefault(decayingStepSizeEnabled, false)

  final def getDecayingStepSizeEnabled: Boolean = $(decayingStepSizeEnabled)
}


private[mixture] trait HasMinibatchSize extends Params {

  final val minibatchSize: Param[Int] = new IntParam(
    this,
    "minibatchSize",
    "minibatchSize",
    (param: Int) => param >= 1
  )

  setDefault(minibatchSize, 1)

  final def getMinibatchSize: Int = $(minibatchSize)

}

private[mixture] trait HasUpdateHoldout extends Params {

  final val updateHoldout: Param[Int] = new IntParam(
    this,
    "updateHoldout",
    "updateHoldout",
    (param: Int) => param >= 1)

  setDefault(updateHoldout, 1)

  final def getUpdateHoldout: Int = $(updateHoldout)
}

/**
 * Param for sample column
 */
private[mixture] trait HasSampleCol extends Params {

  /**
   * Param for measurement column containing measurement vector.
   * @group param
   */
  final val sampleCol: Param[String] = new Param[String](
    this,
    "sampleCol",
    "Column name for samples")

  setDefault(sampleCol, "sample")

  /**
   * Getter for measurement vector column.
   * @group getParam
   */
  final def getSampleCol: String = $(sampleCol)
}