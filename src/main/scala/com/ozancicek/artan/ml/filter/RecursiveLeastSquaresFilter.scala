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

package com.ozancicek.artan.ml.filter

import com.ozancicek.artan.ml.state.{RLSState, RLSUpdate}
import com.ozancicek.artan.ml.state.{StateUpdateFunction, StatefulTransformer}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix, Vector, Vectors, Matrices, Matrix}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{BLAS}
import org.apache.spark.sql._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._


trait HasForgettingFactor extends Params {

  final val forgettingFactor: Param[Double] = new DoubleParam(
    this,
    "forgettingFactor",
    "Forgetting factor",
    ParamValidators.ltEq(1.0))

  setDefault(forgettingFactor, 1.0)

  final def getForgettingFactor: Double = $(forgettingFactor)
}


class RecursiveLeastSquaresFilter(
    val stateSize: Int,
    override val uid: String)
  extends StatefulTransformer[String, RLSUpdate, RLSState]
  with HasGroupKeyCol with HasLabelCol with HasFeaturesCol with HasForgettingFactor
  with HasStateMean with HasStateCovariance {

  implicit val rlsUpdateEncoder = Encoders.product[RLSUpdate]
  implicit val rlsStateEncoder = Encoders.product[RLSState]
  implicit val groupKeyEncoder = Encoders.STRING

  def this(stateSize: Int) = this(stateSize, Identifiable.randomUID("recursiveLeastSquaresFilter"))

  def keyFunc = (in: RLSUpdate) => in.groupKey

  override def copy(extra: ParamMap): RecursiveLeastSquaresFilter = defaultCopy(extra)

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol, "label")

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  setDefault(featuresCol, "features")

  def setGroupKeyCol(value: String): this.type = set(groupKeyCol, value)

  def setForgettingFactor(value: Double): this.type = set(forgettingFactor, value)

  def setInverseCovariance(value: Matrix): this.type = set(stateCov, value)

  def setInverseCovarianceDiag(value: Double): this.type = set(stateCov, getInverseCovMat(value))

  setDefault(
    stateCov, getInverseCovMat(100))

  private def getInverseCovMat(value: Double): DenseMatrix = {
    new DenseMatrix(stateSize, stateSize, DenseMatrix.eye(stateSize).values.map(_ * value))
  }

  private def validateSchema(schema: StructType): Unit = {
    require(isSet(groupKeyCol), "Group key column must be set")
    require(schema($(groupKeyCol)).dataType == StringType, "Group key column must be StringType")
    require(schema($(labelCol)).dataType == DoubleType)
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType)
  }

  def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    rlsUpdateEncoder.schema
  }

  def filter(dataset: Dataset[_]): Dataset[RLSState] = {
    transformSchema(dataset.schema)
    val rlsUpdateDS = dataset
      .withColumn("groupKey", col($(groupKeyCol)))
      .withColumn("label", col($(labelCol)))
      .withColumn("features", col($(featuresCol)))
      .select("groupKey", "label", "features")
      .as(rlsUpdateEncoder)
    transformWithState(rlsUpdateDS)
  }

  def transform(dataset: Dataset[_]): DataFrame = filter(dataset).toDF

  def stateUpdateFunc = new RecursiveLeastSquaresUpdateFunction(
    getStateMean,
    getStateCov,
    getForgettingFactor)
}


private[ml] class RecursiveLeastSquaresUpdateFunction(
    val stateMean: Vector,
    val stateCov: Matrix,
    val forgettingFactor: Double)
  extends StateUpdateFunction[String, RLSUpdate, RLSState] {

  def updateGroupState(
    key: String,
    row: RLSUpdate,
    state: Option[RLSState]): Option[RLSState] = {

    val features = row.features
    val label = row.label

    val currentState = state
      .getOrElse(RLSState(key, 0L, stateMean, stateCov))

    val model = currentState.covariance.transpose.multiply(features)
    val gain = currentState.covariance.multiply(features)
    BLAS.scal(1.0/(forgettingFactor + BLAS.dot(model, features)), gain)
    val residual = label -  BLAS.dot(features, currentState.mean)

    val estMean = currentState.mean.copy
    BLAS.axpy(residual, gain, estMean)
    val gainUpdate = DenseMatrix.zeros(gain.size, features.size)
    BLAS.dger(1.0, gain, features.toDense, gainUpdate)

    val currentCov = currentState.covariance.toDense
    val covUpdate = currentCov.copy
    BLAS.gemm(-1.0, gainUpdate, currentCov, 1.0, covUpdate)

    val estCov = DenseMatrix.zeros(covUpdate.numRows, covUpdate.numCols)
    BLAS.axpy(1.0/forgettingFactor, covUpdate, estCov)

    val newState = RLSState(key, currentState.index + 1L, estMean, estCov)
    Some(newState)
  }
}
