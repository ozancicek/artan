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

import com.github.ozancicek.artan.ml.state.{CategoricalMixtureInput, CategoricalMixtureOutput, CategoricalMixtureState, StatefulTransformer}
import com.github.ozancicek.artan.ml.stats.{CategoricalDistribution, CategoricalMixtureDistribution}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._


/**
 * Online categorical mixture transformer, based on Cappe(2010) Online Expectation-Maximisation
 *
 * @param mixtureCount number of mixture components
 */
class CategoricalMixture(
    val mixtureCount: Int,
    override val uid: String)
  extends StatefulTransformer[
    String,
    CategoricalMixtureInput,
    CategoricalMixtureState,
    CategoricalMixtureOutput,
    CategoricalMixture]
  with HasInitialProbabilities with HasInitialProbabilitiesCol
  with HasCategoricalMixtureModelCol with MixtureParams[CategoricalMixture] {

  protected implicit val stateKeyEncoder = Encoders.STRING

  def this(mixtureCount: Int) = this(mixtureCount, Identifiable.randomUID("CategoricalMixture"))

  protected val defaultStateKey: String = "em.CategoricalMixture.defaultStateKey"

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   */
  override def copy(extra: ParamMap): CategoricalMixture =  {
    val that = new CategoricalMixture(mixtureCount)
    copyValues(that, extra)
  }

  /**
   * Applies the transformation to dataset schemas
   */
  def transformSchema(schema: StructType): StructType = {
    asDataFrameTransformSchema(outEncoder.schema)
  }

  def setInitialProbabilities(value: Array[Array[Double]]): CategoricalMixture = set(initialProbabilities, value)

  def setInitialProbabilitiesCol(value: String): CategoricalMixture = set(initialProbabilitiesCol, value)

  /**
   * Transforms dataset of count to dataframe of estimated states
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val counts = dataset
      .withColumn("sample", col($(sampleCol)))
      .withColumn("stepSize", getUDFWithDefault(stepSize, stepSizeCol))

    val mixtureInput = if (isSet(categoricalMixtureModelCol)) {
      counts.withColumn("initialMixtureModel", col(getCategoricalMixtureModelCol))
    } else {
      val mixtureModelFunc = udf(
        (weights: Seq[Double], probabilities: Seq[Seq[Double]]) =>
          CategoricalMixtureDistribution(weights, probabilities.map(r => CategoricalDistribution(new DenseVector(r.toArray)))))
      val mixtureModelExpr = mixtureModelFunc(col("initialWeights"), col("initialProbabilities"))
      counts
        .withColumn("initialWeights", getUDFWithDefault(initialWeights, initialWeightsCol))
        .withColumn("initialProbabilities", getUDFWithDefault(initialProbabilities, initialProbabilitiesCol))
        .withColumn("initialMixtureModel", mixtureModelExpr)
    }

    asDataFrame(transformWithState(mixtureInput))
  }

  protected def stateUpdateSpec = new MixtureUpdateSpec[
    Int,
    CategoricalDistribution,
    CategoricalMixtureDistribution,
    CategoricalMixtureInput,
    CategoricalMixtureState,
    CategoricalMixtureOutput](getUpdateHoldout, getMinibatchSize)

}


private[mixture] trait HasInitialProbabilities extends Params {

  final val initialProbabilities: Param[Array[Array[Double]]] = new DoubleArrayArrayParam(
    this,
    "initialProbabilities",
    "initialProbabilities")

  final def getInitialMeans: Array[Array[Double]] = $(initialProbabilities)
}


private[mixture] trait HasInitialProbabilitiesCol extends Params {

  final val initialProbabilitiesCol: Param[String] = new Param[String](
    this,
    "initialProbabilitiesCol",
    "initialProbabilitiesCol"
  )

  final def getInitialMeansCol: String = $(initialProbabilitiesCol)
}


private[mixture] trait HasCategoricalMixtureModelCol extends Params {

  final val categoricalMixtureModelCol: Param[String] = new Param[String](
    this,
    "categoricalMixtureModelCol",
    "categoricalMixtureModelCol")

  final def getCategoricalMixtureModelCol: String = $(categoricalMixtureModelCol)
}
