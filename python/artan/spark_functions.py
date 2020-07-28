#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


from pyspark.sql.column import (
    Column, _to_java_column, _to_seq
)
from pyspark import SparkContext
import pyspark.sql.functions as F
import numpy as np


def _sc():
    return SparkContext._active_spark_context


def _spark_functions():
    return _sc()._jvm.com.github.ozancicek.artan.ml.SparkFunctions


def _function_factory(cols, func):
    return Column(func.apply(_to_seq(_sc(), cols, _to_java_column)))


def arrayToVector(arrayCol):
    """
    Converts array of doubles column to dense vector column

    :param arrayCol: Array of doubles column, either string column name or column expression
    :return: DenseVector column
    """
    return _function_factory([arrayCol], _spark_functions().arrayToVector())


def vectorToArray(vectorCol):
    """
    Converts vector column to array of doubles

    :param vectorCol: Vector column, either string column name or column expression
    :return: Array of doubles column
    """
    return _function_factory([vectorCol], _spark_functions().vectorToArray())


def arrayToMatrix(numRowsCol, numColsCol, arrayCol):
    """
    Creates dense matrix from array of doubles column

    :param numRowsCol: Number of rows column, either integer or column expression
    :param numColsCol: Number of cols column, either integer or column expression
    :param arrayCol: Array of doubles column expression, should be column major ordered
    :return: DenseMatrix column
    """
    numRows = F.lit(numRowsCol) if isinstance(numRowsCol, int) else numRowsCol
    numCols = F.lit(numColsCol) if isinstance(numColsCol, int) else numColsCol
    return _function_factory([numRows, numCols, arrayCol], _spark_functions().arrayToMatrix())


def matrixToArray(matrixCol):
    """
    Converts matrix column to array of doubles

    :param matrixCol: Matrix column, either string column name or column expression
    :return: struct of (numRows, numCols, values) where values are column major values of matrix
    """
    return _function_factory([matrixCol], _spark_functions().matrixToArray())


def zerosVector(sizeCol):
    """
    Creates vector of zeros column

    :param sizeCol: Size of the vector, either integer or column expression
    :return: Vector column
    """
    size = F.lit(sizeCol) if isinstance(sizeCol, int) else sizeCol
    return _function_factory([size], _spark_functions().zerosVector())


def onesVector(sizeCol):
    """
    Creates vector of ones column

    :param sizeCol: Size of the vector, either integer or column expression
    :return: Vector column
    """
    size = F.lit(sizeCol) if isinstance(sizeCol, int) else sizeCol
    return _function_factory([size], _spark_functions().onesVector())


def dotVector(leftVectorCol, rightVectorCol):
    """
    Returns the dot product of two vector columns

    :param leftVectorCol: Vector column
    :param rightVectorCol: Vector column
    :return: Double column
    """
    return _function_factory([leftVectorCol, rightVectorCol], _spark_functions().dotVector())


def scalVector(scaleCol, vectorCol):
    """
    Scales the vector column with a scalar constant

    :param scaleCol: scale constant, either numeric or column expression
    :param vectorCol: Vector column
    :return: Vector column
    """
    scale = F.lit(float(scaleCol)) if isinstance(scaleCol, (int, float)) else scaleCol
    return _function_factory([scale, vectorCol], _spark_functions().scalVector())


def axpyVector(alphaCol, xVectorCol, yVectorCol):
    """
    Axpy operation on vectors, alpha*x +y where alpha is scalar, x and y vectors

    :param alphaCol: scalar which scales x, either numeric or column expression
    :param xVectorCol: Vector column corresponding to x
    :param yVectorCol: Vector column corresponding to y
    :return: Vector column
    """
    alpha = F.lit(float(alphaCol)) if isinstance(alphaCol, (int, float)) else alphaCol
    return _function_factory([alpha, xVectorCol, yVectorCol], _spark_functions().axpyVector())


def eyeMatrix(sizeCol):
    """
    Creates identity matrix column

    :param sizeCol: Size of the matrix, either integer or column expression
    :return: DenseMatrix column
    """
    size = F.lit(sizeCol) if isinstance(sizeCol, int) else sizeCol
    return _function_factory([size], _spark_functions().eyeMatrix())


def zerosMatrix(numRowsCol, numColsCol):
    """
    Creates zeros matrix column

    :param numRowsCol: Number of rows column, either integer or column expression
    :param numColsCol: Number of cols column, either integer or column expression
    :return: Matrix column
    """
    numRows = F.lit(numRowsCol) if isinstance(numRowsCol, int) else numRowsCol
    numCols = F.lit(numColsCol) if isinstance(numColsCol, int) else numColsCol
    return _function_factory([numRows, numCols], _spark_functions().zerosMatrix())


def diagMatrix(diagVectorCol):
    """
    Creates a diag matrix from vector

    :param diagVectorCol: Vector column corresponding to diagonal
    :return: Matrix column
    """
    return _function_factory([diagVectorCol], _spark_functions().diagMatrix())


def multiplyMatrix(leftCol, rightCol):
    """
    Matrix multiplication

    :param leftCol: Matrix column
    :param rightCol: Matrix column
    :return: Matrix column for the result of matrix multiplication
    """
    return _function_factory([leftCol, rightCol], _spark_functions().multiplyMatrix())


def multiplyMatrixVector(matCol, vecCol):
    """
    Multiply matrix with a vector

    :param matCol: Matrix column
    :param vecCol: Vector column
    :return: Vector column for the result of matrix multiplication
    """
    return _function_factory([matCol, vecCol], _spark_functions().multiplyMatrixVector())


def projectMatrix(matCol, projectionCol):
    """
    Project matrix operation, i.e A * B * A.T where A is the projection matrix

    :param matCol: Matrix column
    :param projectionCol: Projection matrix column
    :return: Matrix column for the result
    """
    return _function_factory([matCol, projectionCol], _spark_functions().projectMatrix())


def randnMultiGaussian(meanArray, covMatrix, seed=0):
    """
    Samples from multivariate gaussian as vector

    :param meanArray: mean of the distribution, either List[Float] or numpy array
    :param covMatrix: covariance of the distribution, either List[List[Float]]] (row major) or numpy 2d array
    :param seed: seed of the rand
    :return: DenseVector column
    """
    root = np.linalg.cholesky(np.array(covMatrix))
    rows, columns = root.shape

    root = arrayToMatrix(
        F.lit(rows), F.lit(columns),
        F.array([F.lit(el) for el in root.reshape(int(rows*columns), order="F").tolist()]))
    mean = arrayToVector(F.array([F.lit(float(el)) for el in meanArray]))

    samples = arrayToVector(F.array([F.randn(seed=seed+el) for el in range(0, len(meanArray))]))
    return _function_factory([mean, root, samples], _spark_functions().scaleToMultiGaussian())