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

import com.github.ozancicek.artan.ml.state.{MixtureInput, MixtureOutput, MixtureState, MixtureStateFactory, StateUpdateSpec}
import com.github.ozancicek.artan.ml.stats.{Distribution, MixtureDistribution, MixtureDistributionFactory}
import com.github.ozancicek.artan.ml.state.StatefulTransformer
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}
import org.apache.spark.sql.functions.{col, count, lit, max, struct, sum}
import org.apache.spark.sql.types.StructType

import scala.math.{abs, pow}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/**
 * Base trait for common mixture parameters
 *
 * @tparam TransformerType Type of the mixture transformer
 */
private[mixture] trait MixtureParams[TransformerType]
  extends HasInitialWeights with HasInitialWeightsCol with HasStepSizeCol with HasStepSize with HasSampleCol
  with HasUpdateHoldout with HasMinibatchSize with HasMinibatchSizeCol with HasDecayRate with HasInitialMixtureModelCol
  with HasBatchTrainMaxIter with HasBatchTrainTol with HasUpdateHoldoutCol with HasBatchTrainEnabled {

  /**
   * Sets the initial weights of the mixtures. The weights should sum up to 1.0.
   *
   * @group setParam
   */
  def setInitialWeights(value: Array[Double]): TransformerType = set(initialWeights, value)
    .asInstanceOf[TransformerType]

  /**
   * Sets the initial weights of the mixtures from dataframe column. Column should contain array of doubles.
   * Overrides the value set by [[setInitialWeights]].
   *
   * @group setParam
   */
  def setInitialWeightsCol(value: String): TransformerType = set(initialWeightsCol, value)
    .asInstanceOf[TransformerType]

  /**
   * Sets the step size parameter, which weights the current parameter of the model against the old parameter.
   * A step size of 1.0 means ignore the old parameter, whereas a step size of 0 means ignore the current parameter.
   * Values closer to 1.0 will increase speed of convergence, but might have adverse effects on stability. In an online
   * setting, its advised to set it close to 0.0.
   *
   * Default is 0.1
   *
   * @group setParam
   */
  def setStepSize(value: Double): TransformerType = set(stepSize, value).asInstanceOf[TransformerType]

  /**
   * Sets the step size from dataframe column, which would allow setting different step sizes accross measurements.
   * Overrides the value set by [[setStepSize]].
   *
   * @group setParam
   */
  def setStepSizeCol(value: String): TransformerType = set(stepSizeCol, value).asInstanceOf[TransformerType]

  /**
   * Sets the sample column for the mixture model inputs. Depending on the mixture distribution, sample type should
   * be different.
   *
   * Bernoulli => Boolean
   * Poisson => Long
   * MultivariateGaussian => Vector
   *
   * @group setParam
   */
  def setSampleCol(value: String): TransformerType = set(sampleCol, value).asInstanceOf[TransformerType]

  /**
   * Sets the update holdout parameter which controls after how many samples the mixture will start calculating
   * estimates. Preventing update in first few samples might be preferred for stability.
   *
   * @group setParam
   */
  def setUpdateHoldout(value: Int): TransformerType = set(updateHoldout, value).asInstanceOf[TransformerType]

  /**
   * Sets the update holdout parameter from dataframe column rather than a constant value across all states.
   * Overrides the value set by [[setUpdateHoldout]].
   *
   * @group setParam
   */
  def setUpdateHoldoutCol(value: String): TransformerType = set(updateHoldoutCol, value)
    .asInstanceOf[TransformerType]

  /**
   * Sets the minibatch size for batching samples together in online EM algorithm. Estimate will be produced once
   * per each batch. Having larger batches increases stability with increased memory footprint.
   *
   * Default is 1
   *
   * @group setParam
   */
  def setMinibatchSize(value: Int): TransformerType = set(minibatchSize, value).asInstanceOf[TransformerType]

  /**
   * Sets the minibatch size from dataframe column rather than a constant minibatch size across all states.
   * Overrides [[setMinibatchSize]] setting.
   *
   * @group setParam
   */
  def setMinibatchSizeCol(value: String): TransformerType = set(minibatchSizeCol, value).asInstanceOf[TransformerType]

  /**
   * Sets the step size as a decaying function rather than a constant step size, which might be preferred
   * for batch training. If set, the step size will be replaced with the output of following function:
   *
   * stepSize = pow(2 + kIter, -decayRate)
   *
   * Where kIter is incremented by 1 at each minibatch.
   *
   * @group setParam
   */
  def setDecayRate(value: Double): TransformerType = set(decayRate, value).asInstanceOf[TransformerType]


  /**
   * Sets the initial mixture model directly from dataframe column
   *
   * @group setParam
   */
  def setInitialMixtureModelCol(value: String): TransformerType = set(initialMixtureModelCol, value)
    .asInstanceOf[TransformerType]

  /**
   * Sets the maximum iterations for batch EM mode
   *
   * @group setParam
   */
  def setBatchTrainMaxIter(value: Int): TransformerType = set(batchTrainMaxIter, value)
    .asInstanceOf[TransformerType]


  /**
   * Sets the stopping criteria in terms of loglikelihood improvement for batch EM mode
   *
   * @group setParam
   */
  def setBatchTrainTol(value: Double): TransformerType = set(batchTrainTol, value)
    .asInstanceOf[TransformerType]

  /**
   * Enables batch EM mode. When enabled, [[transform]] method will do an iterative EM training with multiple passes
   * as opposed to online training with single pass.
   *
   * Disabled by default
   *
   * @group setParam
   */
  def setEnableBatchEM: TransformerType = set(batchTrainEnabled, true).asInstanceOf[TransformerType]
}


private[mixture] abstract class FiniteMixture[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType]: TypeTag,
  InputType <: MixtureInput[SampleType, DistributionType, MixtureType] : TypeTag : Manifest,
  StateType <: MixtureState[SampleType, DistributionType, MixtureType] : ClassTag,
  OutputType <: MixtureOutput[SampleType, DistributionType, MixtureType] : TypeTag,
  TransformerType <: FiniteMixture[
    SampleType, DistributionType, MixtureType, InputType, StateType, OutputType, TransformerType]](implicit
  stateFactory: MixtureStateFactory[
    SampleType, DistributionType, MixtureType, StateType, OutputType],
  mixtureFactory: MixtureDistributionFactory[
    SampleType, DistributionType, MixtureType])
  extends StatefulTransformer[String, InputType, StateType, OutputType, TransformerType]
  with MixtureParams[TransformerType] {

  protected implicit val stateKeyEncoder = Encoders.STRING

  private def mixtureSchema: StructType = Encoders.product[MixtureType].schema

  protected def transformAndValidateSchema(schema: StructType): StructType = {
    if (isSet(initialMixtureModelCol)) {
      val inSchema = schema(getInitialMixtureModelCol).dataType
      require(inSchema == mixtureSchema,
        s"Schema of initial mixture model $inSchema doesn't match type $mixtureSchema")
    }
    transformSchema(schema)
  }

  /**
   * Convert samples dataframe to mixture input dataframe
   */
  private def asMixtureInput(dataset: Dataset[_]): DataFrame = {
    val samples = dataset
      .withColumn("sample", col($(sampleCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))
      .withColumn("decayRate", getDecayRateExpr())
      .withColumn("minibatchSize", getUDFWithDefault(minibatchSize, minibatchSizeCol))
      .withColumn("updateHoldout", getUDFWithDefault(updateHoldout, updateHoldoutCol))
    val mixtureInput = if (isSet(initialMixtureModelCol)) {
      samples.withColumn("initialMixtureModel", col(getInitialMixtureModelCol))
    } else {
      buildInitialMixtureModel(samples)
    }
    mixtureInput
  }

  /**
   * Build the initialMixtureModel column from distribution specific parameters
   */
  protected def buildInitialMixtureModel(dataframe: DataFrame): DataFrame


  /**
   * Transforms the dataframe of samples to a dataframe of mixture parameter estimates.
   */
  def transform(dataset: Dataset[_]): DataFrame = if (getBatchTrainEnabled) batchEM(dataset) else onlineEM(dataset)

  /**
   * Online EM, mixture model estimation with a single pass.
   *
   * Valid for both streaming datasets and batch datasets.
   */
  private def onlineEM(dataset: Dataset[_]): DataFrame = {
    transformAndValidateSchema(dataset.schema)

    val mixtureInput = asMixtureInput(dataset)
    asDataFrame(transformWithState(mixtureInput))
  }

  /**
   * Batch EM, mixture model estimation with multiple passes. Note that implementation is not optimized
   * for performance, it re-uses the same procedures designed for Online EM. It can be used for creating
   * a prior for online algorithm, or offline refinement of online results for a subset of samples.
   *
   * Only valid for batch datasets.
   */
  private def batchEM(dataset: Dataset[_]): DataFrame = {
    transformAndValidateSchema(dataset.schema)

    val maxIter = getBatchTrainMaxIter
    val minDelta = getBatchTrainTol

    log.info(s"Starting EM iterations with min loglikelihood improvement $minDelta and maxIter $maxIter")

    // Ensure minibatchSize = sampleSize, which will
    // convert stochastic expectation approximation to 'traditional' expectation
    val samples = asMixtureInput(dataset)
      .withColumn("stateKey", getStateKeyColumn)
      .withColumn("decayRate", lit(null))
      .withColumn("updateHoldout", lit(0))
      .drop("minibatchSize").localCheckpoint()

    // Calculate minibatch size across states and join with samples
    val minibatchSize = samples.groupBy("stateKey")
      .agg(count("sample").alias("minibatchSize"))
    val mixtureInput = samples.join(minibatchSize, Seq("stateKey"), "left").localCheckpoint()

    val emIter = (in: DataFrame) => {
      val modelState = struct(col("stateIndex"), col("mixtureModel"), col("loglikelihood"))
      asDataFrame(transformWithState(in))
        .withColumn("modelState", modelState)
        .groupBy("stateKey").agg(max("modelState").alias("modelState"))
        .select(
          col("stateKey"),
          col("modelState.mixtureModel").alias("initialMixtureModel"),
          col("modelState.loglikelihood"))
    }

    val loglikelihood = (in: DataFrame) => {
      in.select(sum("loglikelihood")).head.getAs[Double](0)
    }

    var model = emIter(mixtureInput)
      .localCheckpoint()

    var ll = loglikelihood(model)
    var delta = Double.MaxValue - abs(ll)
    var iteration = 1

    log.info(s"Initial model likelihood $ll")
    while (iteration < maxIter & delta > minDelta) {
      val input = mixtureInput.drop("initialMixtureModel").join(model, Seq("stateKey"))
      model = emIter(input).localCheckpoint()

      val currentLikelihood = loglikelihood(model)
      delta = currentLikelihood - ll
      ll = currentLikelihood
      iteration = iteration + 1

      log.info(s"Iteration $iteration loglikelihood $ll improvement $delta")
      if (delta < 0.0) {
        log.warn(s"Loglikelihood decreased on iteration $iteration")
      }
    }
    model.withColumnRenamed("initialMixtureModel", "mixtureModel")
  }

  protected def stateUpdateSpec = new MixtureUpdateSpec[
    SampleType,
    DistributionType,
    MixtureType,
    InputType,
    StateType,
    OutputType](getUpdateHoldout)(stateFactory, mixtureFactory)
}

/**
 * State update spec of online EM for finite mixtures.
 */
private[mixture] class MixtureUpdateSpec[
  SampleType,
  DistributionType <: Distribution[SampleType, DistributionType],
  MixtureType <: MixtureDistribution[SampleType, DistributionType, MixtureType],
  InputType <: MixtureInput[SampleType, DistributionType, MixtureType],
  StateType <: MixtureState[SampleType, DistributionType, MixtureType],
  OutputType <: MixtureOutput[SampleType, DistributionType, MixtureType]](
  updateHoldout: Int)(implicit
  stateFactory: MixtureStateFactory[
    SampleType, DistributionType, MixtureType, StateType, OutputType],
  mixtureFactory: MixtureDistributionFactory[
    SampleType, DistributionType, MixtureType]) extends StateUpdateSpec[String, InputType, StateType, OutputType]{

  /**
   * Convert mixture state to output
   */
  protected def stateToOutput(
    key: String,
    row: InputType,
    state: StateType): List[OutputType] = {
    // Output once per minibatch, which is tracked from #samples stored in state
    if (state.samples.isEmpty) {
      List(stateFactory.createOutput(
        key,
        state.stateIndex,
        state.mixtureModel,
        row.eventTime,
        state.loglikelihood))
    } else {
      List.empty[OutputType]
    }
  }

  /**
   * Function to create initial state for all mixtures
   */
  private def getInitialState(row: InputType): StateType = {
    val weightedDist = MixtureDistribution.weightedMixture[SampleType, DistributionType, MixtureType](
      row.initialMixtureModel)
    stateFactory.createState(
      0L,
      List.empty[SampleType],
      weightedDist,
      row.initialMixtureModel,
      Double.MinValue
    )
  }

  /**
   * Function to progress state
   */
  private def calculateNextState(
    row: InputType,
    state: Option[StateType]): Option[StateType] = {
    val currentState = state.getOrElse(getInitialState(row))
    val newSamples = row.sample :: currentState.samples

    // Only push samples until samplesSize < minibatchSize, and progress state when samplesSize = minibatchSize
    val nextState = if (newSamples.size < row.minibatchSize) {
      stateFactory.createState(
        currentState.stateIndex, newSamples, currentState.summaryModel, currentState.mixtureModel, Double.MinValue)
    } else {

      // Determine the step size, which is the inertia <1.0 of the sufficient statistics of current vs previous state.
      val stepSize = row.decayRate match {
        case Some(rate) => pow(2 + currentState.stateIndex, -rate)
        case None => row.stepSize
      }

      // Stochastic update of sufficient statistics
      val newSummaryModel = MixtureDistribution
        .stochasticUpdate[SampleType, DistributionType, MixtureType](
        currentState.summaryModel, currentState.mixtureModel, newSamples, stepSize)

      // Calculate mixture params from sufficient statistics
      val newMixtureModel = if (currentState.stateIndex < updateHoldout) {
        currentState.mixtureModel
      } else {
        MixtureDistribution.inverseWeightedMixture[SampleType, DistributionType, MixtureType](newSummaryModel)
      }

      // Calculate the loglikelihood of new parameters
      val ll = newMixtureModel.loglikelihood(newSamples)
      val stateIndex = currentState.stateIndex + 1
      stateFactory.createState(stateIndex, List.empty[SampleType], newSummaryModel, newMixtureModel, ll)
    }
    Some(nextState)
  }

  protected def updateGroupState(
    key: String,
    row: InputType,
    state: Option[StateType]): Option[StateType] = calculateNextState(row, state)

}