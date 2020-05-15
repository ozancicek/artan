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

import com.github.ozancicek.artan.ml.state.{GaussianMixtureInput, GaussianMixtureOutput, GaussianMixtureState}
import com.github.ozancicek.artan.ml.stats.{GaussianMixtureDistribution, MultivariateGaussianDistribution}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
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
  extends FiniteMixture[
    Vector,
    MultivariateGaussianDistribution,
    GaussianMixtureDistribution,
    GaussianMixtureInput,
    GaussianMixtureState,
    GaussianMixtureOutput,
    MultivariateGaussianMixture]
  with HasInitialMeans with HasInitialMeansCol
  with HasInitialCovariances with HasInitialCovariancesCol {

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

  def setInitialMeansCol(value: String): MultivariateGaussianMixture = {
    set(initialMeansCol, value)
  }

  def setInitialCovariances(value: Array[Array[Double]]): MultivariateGaussianMixture = {
    set(initialCovariances, value)
  }

  def setInitialCovariancesCol(value: String): MultivariateGaussianMixture = {
    set(initialCovariancesCol, value)
  }

  /**
   * Applies the transformation to dataset schema
   */
  def transformSchema(schema: StructType): StructType = {
    if (!isSet(initialMixtureModelCol)) {
      require(
        isSet(initialMeans) | isSet(initialMeansCol), "Initial means or its dataframe column must be set")
      require(
        isSet(initialCovariances) | isSet(initialCovariancesCol), "Initial covariances or its columns must be set")

      if (isSet(initialMeansCol)) {
        require(
          schema(getInitialMeansCol).dataType == ArrayType(ArrayType(DoubleType)),
          "Initial means column should be a nested array of doubles with dimensions mixtureCount x featureSize")
      }
      if (isSet(initialCovariancesCol)) {
        require(
          schema(getInitialCovariancesCol).dataType == ArrayType(ArrayType(DoubleType)),
          "Initial covariances column should be a nested array of doubles with dimensions mixtureCount x featureSize^2"
        )
      }
    }
    asDataFrameTransformSchema(outEncoder.schema)
  }

  protected def buildInitialMixtureModel(dataFrame: DataFrame): DataFrame = {
    val mixtureModelFunc = udf(
      (weights: Seq[Double], means: Seq[Seq[Double]], covariances: Seq[Seq[Double]]) => {
        val gaussians = means.zip(covariances).map { s =>
          val meanVector = new DenseVector(s._1.toArray)
          val covMatrix = new DenseMatrix(s._1.size, s._1.size, s._2.toArray)
          MultivariateGaussianDistribution(meanVector, covMatrix)
        }
        GaussianMixtureDistribution(weights, gaussians)
      }
    )
    val mixtureModelExpr = mixtureModelFunc(
      col("initialWeights"),
      col("initialMeans"),
      col("initialCovariances"))

    dataFrame
      .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
      .withColumn("initialMeans", getUDFWithDefault(initialMeans, initialMeansCol))
      .withColumn("initialCovariances", getUDFWithDefault(initialCovariances, initialCovariancesCol))
      .withColumn("initialMixtureModel", mixtureModelExpr)
      .drop("initialWeights", "initialMeans", "initialCovariances")
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