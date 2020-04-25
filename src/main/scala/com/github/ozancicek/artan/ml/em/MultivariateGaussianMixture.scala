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

import com.github.ozancicek.artan.ml.state.{GaussianMixtureInput, GaussianMixtureOutput}
import com.github.ozancicek.artan.ml.state.{GaussianMixtureModel, GaussianMixtureState}
import com.github.ozancicek.artan.ml.state.{StateUpdateSpec, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.MultivariateGaussianDistribution
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Experimental online multivariate gaussian mixture transformer, based on Cappe(2010) Online Expectation-Maximisation
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
  with HasGaussianMixtureModelCol with HasStepSize with HasStepSizeCol with HasMeasurementCol
  with HasInitialWeights with HasInitialWeightsCol with HasInitialMeans with HasInitialMeansCol
  with HasInitialCovariances with HasInitialCovariancesCol with HasMeasurementSize {

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

  def setInitialWeights(value: Array[Double]): MultivariateGaussianMixture = {
    set(initialWeights, value)
  }

  def setInitialWeightsCol(value: String): MultivariateGaussianMixture = {
    set(initialWeightsCol, value)
  }

  def setStepSize(value: Double): MultivariateGaussianMixture = {
    set(stepSize, value)
  }

  def setStepSizeCol(value: String): MultivariateGaussianMixture = {
    set(stepSizeCol, value)
  }

  def setInitialMeans(value: Array[Array[Double]]): MultivariateGaussianMixture = {
    set(initialMeans, value)
  }

  def setInitialCovariances(value: Array[Array[Double]]): MultivariateGaussianMixture = {
    set(initialCovariances, value)
  }

  def setMeasurementSize(value: Int): MultivariateGaussianMixture = {
    set(measurementSize, value)
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
      .withColumn("measurement", col($(measurementCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))

    val mixtureInput = if (isSet(gaussianMixtureModelCol)) {
      measurements.withColumn("initialMixtureModel", col(getGaussianMixtureModelCol))
    } else {
      val mixtureModelFunc = udf(
        (weights: Seq[Double], means: Seq[Seq[Double]], covariances: Seq[Seq[Double]]) => {
          val gaussians = means.zip(covariances).map { s =>
            val meanVector = new DenseVector(s._1.toArray)
            val covMatrix = new DenseMatrix(getMeasurementSize, getMeasurementSize, s._2.toArray)
            MultivariateGaussianDistribution(meanVector, covMatrix)
          }
          GaussianMixtureModel(weights.toArray, gaussians.toArray)
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

  protected def stateUpdateSpec: GaussianMixtureUpdateSpec = new GaussianMixtureUpdateSpec(5)

}

private[em] class GaussianMixtureUpdateSpec(updateHoldout: Int)
  extends StateUpdateSpec[String, GaussianMixtureInput, GaussianMixtureState, GaussianMixtureOutput] {

  protected def stateToOutput(
    key: String,
    row: GaussianMixtureInput,
    state: GaussianMixtureState): List[GaussianMixtureOutput] = {
    List(GaussianMixtureOutput(
      key,
      state.stateIndex,
      state.mixtureModel,
      row.eventTime))
  }

  def updateGroupState(
    key: String,
    row: GaussianMixtureInput,
    state: Option[GaussianMixtureState]): Option[GaussianMixtureState] = {

    val getInitialState = () => {
      val (ms, cs) = row.initialMixtureModel.distributions
        .zip(row.initialMixtureModel.weights)
        .map { case (dist, weight) =>
          val meanSummary = dist.mean.toDense.copy
          BLAS.scal(weight, meanSummary)
          val covSummary = DenseMatrix.zeros(dist.mean.size, dist.mean.size)
          BLAS.axpy(weight, dist.covariance.toDense, covSummary)
          (meanSummary, covSummary)
        }.unzip

      GaussianMixtureState(0L, row.initialMixtureModel.weights, ms, cs, row.initialMixtureModel)
    }

    val currentState = state
      .getOrElse(getInitialState())

    val likelihood = currentState.mixtureModel.distributions.zip(currentState.mixtureModel.weights).map {
      case (dist, weight) => dist.pdf(row.measurement.toDense) * weight
    }

    val sumLikelihood = likelihood.sum
    val likelihoodWeights = likelihood.map(_ * row.stepSize/sumLikelihood)

    val weightsSummary = currentState.weightsSummary
      .zip(likelihoodWeights).map(s => (1 - row.stepSize) * s._1 + s._2)

    val meansSummary = currentState.meansSummary.zip(likelihoodWeights).map { case(ms, w) =>
      val meanSummary = ms.copy
      BLAS.scal(1 - row.stepSize, meanSummary)
      BLAS.axpy(w, row.measurement, meanSummary)
      meanSummary
    }

    val means = currentState.mixtureModel.distributions.map(_.mean)

    val covsSummary = currentState.covariancesSummary.zip(means).zip(likelihoodWeights).map { case((cs, m), w) =>
      val covSummary = DenseMatrix.zeros(cs.numRows, cs.numCols)
      BLAS.axpy(1 - row.stepSize, cs, covSummary)

      val residual = row.measurement.toDense.copy
      BLAS.axpy(-1.0, m, residual)
      BLAS.dger(w, residual, residual, covSummary)
      covSummary
    }

    val newModel = if (currentState.stateIndex < updateHoldout) {
      currentState.mixtureModel
    } else {
      val weights = weightsSummary
      val gaussians = meansSummary.zip(covsSummary).zip(weights).map { case ((ms, cs), w) =>
        val mean = ms.copy
        BLAS.scal(1.0 / w, mean)
        val cov = DenseMatrix.zeros(ms.size, ms.size)
        BLAS.axpy(1.0/w, cs, cov)
        MultivariateGaussianDistribution(mean, cov)
      }
      GaussianMixtureModel(weights, gaussians)
    }
    val nextState = GaussianMixtureState(
      currentState.stateIndex + 1,
      weightsSummary, meansSummary,
      covsSummary,
      newModel)
    Some(nextState)
  }
}


private[em] trait HasInitialMeans extends Params {

  final val initialMeans: Param[Array[Array[Double]]] = new DoubleArrayArrayParam(
    this,
    "initialMeans",
    "initialMeans")

  final def getInitialMeans: Array[Array[Double]] = $(initialMeans)
}

private[em] trait HasInitialMeansCol extends Params {

  final val initialMeansCol: Param[String] = new Param[String](
    this,
    "initialMeansCol",
    "initialMeansCol"
  )

  final def getInitialMeansCol: String = $(initialMeansCol)
}

private[em] trait HasInitialCovariances extends Params {

  final val initialCovariances: Param[Array[Array[Double]]] = new DoubleArrayArrayParam(
    this,
    "initialCovariances",
    "initialCovariances"
  )

  final def getInitialCovariances: Array[Array[Double]] = $(initialCovariances)
}


private[em] trait HasInitialCovariancesCol extends Params {

  final val initialCovariancesCol: Param[String] = new Param[String](
    this,
    "initialCovariancesCol",
    "initialCovariancesCol"
  )

  final def getInitialCovariancesCol: String = $(initialCovariancesCol)
}

private[em] trait HasMeasurementSize extends Params {

  final val measurementSize: Param[Int] = new IntParam(
    this,
    "measurementSize",
    "measurementSize"
  )

  final def getMeasurementSize: Int = $(measurementSize)
}

private[em] trait HasGaussianMixtureModelCol extends Params {

  final val gaussianMixtureModelCol: Param[String] = new Param[String](
    this,
    "gaussianMixtureModelCol",
    "gaussianMixtureModelCol")

  final def getGaussianMixtureModelCol: String = $(gaussianMixtureModelCol)
}

/**
 * Param for measurement column
 */
private[artan] trait HasMeasurementCol extends Params {

  /**
   * Param for measurement column containing measurement vector.
   * @group param
   */
  final val measurementCol: Param[String] = new Param[String](
    this,
    "measurementCol",
    "Column name for measurement vector. Missing measurements are allowed with nulls in the data")

  setDefault(measurementCol, "measurement")

  /**
   * Getter for measurement vector column.
   * @group getParam
   */
  final def getMeasurementCol: String = $(measurementCol)
}