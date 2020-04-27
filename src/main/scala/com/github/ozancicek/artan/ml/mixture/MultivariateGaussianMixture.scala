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

import com.github.ozancicek.artan.ml.state.{GaussianMixtureInput, GaussianMixtureOutput, GaussianMixtureState, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.{GaussianMixtureDistribution, MultivariateGaussianDistribution}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Online multivariate gaussian mixture transformer, based on Cappe(2010) Online Expectation-Maximisation
 *
 * @param mixtureCount number of mixture components
 */
class MultivariateGaussianMixture(
    val mixtureCount: Int,
    override val uid: String)
  extends StatefulTransformer[
    String,
    GaussianMixtureInput,
    GaussianMixtureState,
    GaussianMixtureOutput,
    MultivariateGaussianMixture]
  with HasGaussianMixtureModelCol with HasInitialMeans with HasInitialMeansCol
  with HasInitialCovariances with HasInitialCovariancesCol with HasSampleSize
  with MixtureParams[MultivariateGaussianMixture] {

  protected implicit val stateKeyEncoder = Encoders.STRING

  def this(mixtureCount: Int) = this(mixtureCount, Identifiable.randomUID("MultivariateGaussianMixture"))

  protected val defaultStateKey: String = "em.MultivariateGaussianMixture.defaultStateKey"

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): MultivariateGaussianMixture =  {
    val that = new MultivariateGaussianMixture(mixtureCount)
    copyValues(that, extra)
  }

  def setInitialMeans(value: Array[Array[Double]]): MultivariateGaussianMixture = {
    set(initialMeans, value)
  }

  def setInitialCovariances(value: Array[Array[Double]]): MultivariateGaussianMixture = {
    set(initialCovariances, value)
  }

  def setSampleSize(value: Int): MultivariateGaussianMixture = {
    set(sampleSize, value)
  }
  /**
   * Applies the transformation to dataset schema
   */
  def transformSchema(schema: StructType): StructType = {
    asDataFrameTransformSchema(outEncoder.schema)
  }

  /**
   * Transforms dataset of count to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val measurements = dataset
      .withColumn("sample", col($(sampleCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))

    val mixtureInput = if (isSet(gaussianMixtureModelCol)) {
      measurements.withColumn("initialMixtureModel", col(getGaussianMixtureModelCol))
    } else {
      val mixtureModelFunc = udf(
        (weights: Seq[Double], means: Seq[Seq[Double]], covariances: Seq[Seq[Double]]) => {
          val gaussians = means.zip(covariances).map { s =>
            val meanVector = new DenseVector(s._1.toArray)
            val covMatrix = new DenseMatrix(getSampleSize, getSampleSize, s._2.toArray)
            MultivariateGaussianDistribution(meanVector, covMatrix)
          }
          GaussianMixtureDistribution(weights, gaussians)
        }
      )

      val mixtureModelExpr = mixtureModelFunc(
        col("initialWeights"),
        col("initialMeans"),
        col("initialCovariances"))

      measurements
        .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
        .withColumn("initialMeans", getUDFWithDefault(initialMeans, initialMeansCol))
        .withColumn("initialCovariances", getUDFWithDefault(initialCovariances, initialCovariancesCol))
        .withColumn("initialMixtureModel", mixtureModelExpr)
    }

    asDataFrame(transformWithState(mixtureInput))
  }

  protected def stateUpdateSpec: GaussianMixtureUpdateSpec = new GaussianMixtureUpdateSpec(
    getUpdateHoldout, getMinibatchSize)

}


private[mixture] class GaussianMixtureUpdateSpec(val updateHoldout: Int, val minibatchSize: Int)
  extends MixtureUpdateSpec[
    Vector,
    MultivariateGaussianDistribution,
    GaussianMixtureDistribution,
    GaussianMixtureInput,
    GaussianMixtureState,
    GaussianMixtureOutput] {

  protected def stateToOutput(
    key: String,
    row: GaussianMixtureInput,
    state: GaussianMixtureState): List[GaussianMixtureOutput] = {
    if (state.samples.isEmpty) {
      List(GaussianMixtureOutput(
        key,
        state.stateIndex,
        state.mixtureModel,
        row.eventTime))
    } else {
      List.empty[GaussianMixtureOutput]
    }
  }

  protected def getInitialState(row: GaussianMixtureInput): GaussianMixtureState = {
    GaussianMixtureState(0L, List.empty[Vector], row.initialMixtureModel.weightedDistributions, row.initialMixtureModel)
  }

  protected def updateGroupState(
    key: String,
    row: GaussianMixtureInput,
    state: Option[GaussianMixtureState]): Option[GaussianMixtureState] = {
    val (ind, samples, summary, model) = calculateNextState(row, state)
    Some(GaussianMixtureState(ind, samples, summary, model))
  }
}


private[mixture] trait HasInitialMeans extends Params {

  final val initialMeans: Param[Array[Array[Double]]] = new DoubleArrayArrayParam(
    this,
    "initialMeans",
    "initialMeans")

  final def getInitialMeans: Array[Array[Double]] = $(initialMeans)
}

private[mixture] trait HasInitialMeansCol extends Params {

  final val initialMeansCol: Param[String] = new Param[String](
    this,
    "initialMeansCol",
    "initialMeansCol"
  )

  final def getInitialMeansCol: String = $(initialMeansCol)
}

private[mixture] trait HasInitialCovariances extends Params {

  final val initialCovariances: Param[Array[Array[Double]]] = new DoubleArrayArrayParam(
    this,
    "initialCovariances",
    "initialCovariances"
  )

  final def getInitialCovariances: Array[Array[Double]] = $(initialCovariances)
}


private[mixture] trait HasInitialCovariancesCol extends Params {

  final val initialCovariancesCol: Param[String] = new Param[String](
    this,
    "initialCovariancesCol",
    "initialCovariancesCol"
  )

  final def getInitialCovariancesCol: String = $(initialCovariancesCol)
}

private[mixture] trait HasSampleSize extends Params {

  final val sampleSize: Param[Int] = new IntParam(
    this,
    "sampleSize",
    "sampleSize"
  )

  final def getSampleSize: Int = $(sampleSize)
}

private[mixture] trait HasGaussianMixtureModelCol extends Params {

  final val gaussianMixtureModelCol: Param[String] = new Param[String](
    this,
    "gaussianMixtureModelCol",
    "gaussianMixtureModelCol")

  final def getGaussianMixtureModelCol: String = $(gaussianMixtureModelCol)
}