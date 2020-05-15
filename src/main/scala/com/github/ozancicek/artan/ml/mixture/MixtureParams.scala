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
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf


private[mixture] trait HasInitialWeights extends Params {

  def mixtureCount: Int

  /**
   * Initial weight of the mixtures
   *
   * @group param
   */
  final val initialWeights: Param[Array[Double]] = new DoubleArrayParam(
    this,
    "initialWeights",
    "Initial weights of the mixtures. The weights should sum up to 1.0.")

  setDefault(initialWeights, Array.fill(mixtureCount) {1.0/mixtureCount})

  /**
   * Getter for initialWeights parameter
   *
   * @group getParam
   */
  final def getInitialWeights: Array[Double] = $(initialWeights)
}


private[mixture] trait HasInitialWeightsCol extends Params {

  /**
   * Initial weights as dataframe column
   *
   * @group param
   */
  final val initialWeightsCol: Param[String] = new Param[String](
    this,
    "initialWeightsCol",
    "Initial weights of the mixtures from dataframe column."
  )

  /**
   * Getter for initialWeightsCol parameter
   *
   * @group getParam
   */
  final def getInitialWeightsCol: String = $(initialWeightsCol)
}

private[mixture] trait HasStepSize extends Params {

  /**
   * Controls the inertia of the current parameter.
   *
   * @group param
   */
  final val stepSize: Param[Double] = new DoubleParam(
    this,
    "stepSize",
    "weights the current parameter of the model against the old parameter. A step size of 1.0 means ignore" +
      "the old parameter, whereas a step size of 0 means ignore the current parameter. Values closer to 1.0 will" +
      "increase speed of convergence, but might have adverse effects on stability. In online setting," +
      "its advised to set it close to 0.0.",
    ParamValidators.ltEq(1.0)
  )

  setDefault(stepSize, 0.01)

  /**
   * Getter for stepSize parameter
   *
   * @group getParam
   */
  final def getStepSize: Double = $(stepSize)
}


private[mixture] trait HasStepSizeCol extends Params {

  /**
   * stepSize as dataframe column
   *
   * @group param
   */
  final val stepSizeCol: Param[String] = new Param[String](
    this,
    "stepSizeCol",
    "Sets the stepSize parameter from dataframe column instead of a constant value across all samples")

  /**
   * Getter for stepSizeCol parameter
   *
   * @group getParam
   */
  final def getStepSizeCol: String = $(stepSizeCol)
}


private[mixture] trait HasDecayRate extends Params {

  /**
   * Decaying stepSize
   *
   * @group param
   */
  final val decayRate: Param[Double] = new DoubleParam(
    this,
    "decayRate",
    "Step size as a decaying function rather than a constant, which might be preferred at batch training." +
      "If set, the step size will be replaced with the output of following function" +
      "stepSize = pow(2 + kIter, -decayRate)",
    ParamValidators.ltEq(1.0)
  )

  /**
   * Get decay rate as udf
   *
   * @group getParam
   */
  protected def getDecayRateExpr: UserDefinedFunction = udf(() => get(decayRate))
}


private[mixture] trait HasMinibatchSize extends Params {

  /**
   * Number of samples in a batch
   *
   * @group param
   */
  final val minibatchSize: Param[Int] = new IntParam(
    this,
    "minibatchSize",
    "Size for batching samples together in online EM algorithm. Estimate will be produced once per each batch" +
      "Having larger batches increases stability with increased memory footprint. Each minibatch is stored in" +
    "the state of the mixture transformer, therefore it's independent from spark minibatches.",
    (param: Int) => param >= 1
  )

  setDefault(minibatchSize, 1)

  /**
   * Minibatch param getter
   *
   * @group getParam
   */
  final def getMinibatchSize: Int = $(minibatchSize)

}


private[mixture] trait HasMinibatchSizeCol extends Params {

  /**
   * Number of samples in a batch from dataframe column
   *
   * @group param
   */
  final val minibatchSizeCol: Param[String] = new Param[String](
    this,
    "minibatchSizeCol",
    "Minibatch size from dataframe column to allow different minibatch sizes across states"
  )

  /**
   * MinibatchSizeCol param getter
   *
   * @group getParam
   */
  final def getMinibatchSizeCol: String = $(minibatchSizeCol)

}

private[mixture] trait HasUpdateHoldout extends Params {

  /**
   * Update holdout parameter
   *
   * @group param
   */
  final val updateHoldout: Param[Int] = new IntParam(
    this,
    "updateHoldout",
    "Controls after how many samples the mixture will start calculating estimates. Preventing update" +
      "in first few samples might be preferred for stability.",
    (param: Int) => param >= 0)

  setDefault(updateHoldout, 0)

  /**
   * Getter for update holdout param
   *
   * @group getParam
   */
  final def getUpdateHoldout: Int = $(updateHoldout)
}


private[mixture] trait HasUpdateHoldoutCol extends Params {

  /**
   * Update holdout parameter from dataframe column
   *
   * @group param
   */
  final val updateHoldoutCol: Param[String] = new Param[String](
    this,
    "updateHoldoutCol",
    "updateHoldout from dataframe column rather than a constant value across all states")

  /**
   * Getter for update holdout column param
   *
   * @group getParam
   */
  final def getUpdateHoldoutCol: String = $(updateHoldoutCol)
}


private[mixture] trait HasSampleCol extends Params {

  /**
   * Param for sample column.
   *
   * @group param
   */
  final val sampleCol: Param[String] = new Param[String](
    this,
    "sampleCol",
    "Column name for input to mixture models")

  setDefault(sampleCol, "sample")

  /**
   * Getter for sample column.
   *
   * @group getParam
   */
  final def getSampleCol: String = $(sampleCol)
}


private[mixture] trait HasInitialMixtureModelCol extends Params {

  /**
   * Initial mixture model as a struct column
   *
   * @group param
   */
  final val initialMixtureModelCol: Param[String] = new Param[String](
    this,
    "initialMixtureModelCol",
    "Sets the initial mixture model from struct column conforming to mixture distributions case class")

  /**
   * Getter for initial mixture model column
   *
   * @group getParam
   */
  final def getInitialMixtureModelCol: String = $(initialMixtureModelCol)
}


private[mixture] trait HasBatchTrainMaxIter extends Params {

  /**
   * maxIter for batch EM
   *
   * @group param
   */
  final val batchTrainMaxIter: Param[Int] = new IntParam(
    this,
    "batchTrainMaxIter",
    "Maximum iterations in batch train mode, default is 30",
    ParamValidators.gt(0)
  )
  setDefault(batchTrainMaxIter, 30)

  /**
   * Getter for batch train max iter parameter
   *
   * @group getParam
   */
  final def getBatchTrainMaxIter: Int = $(batchTrainMaxIter)
}


private[mixture] trait HasBatchTrainTol extends Params {

  /**
   * Tolerance to stop iterations in batch EM
   *
   * @group param
   */
  final val batchTrainTol: Param[Double] = new DoubleParam(
    this,
    "batchTrainTol",
    "Stop iteration criteria defined by change in the loglikelihood, default is 0.1",
    ParamValidators.gt(0.0)
  )

  setDefault(batchTrainTol, 0.1)

  final def getBatchTrainTol: Double  = $(batchTrainTol)
}