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


from artan.testing.sql_utils import ReusedSparkTestCase
from artan.spark_functions import (
    arrayToVector, vectorToArray, onesVector, zerosVector, arrayToMatrix, matrixToArray
)
from pyspark.ml.linalg import Vectors, Matrices
import pyspark.sql.functions as F
import numpy as np


class SparkFunctionTests(ReusedSparkTestCase):

    def test_array_vector(self):
        arr = np.array([1.0, 2.0, 3.0])
        vec = self.spark.createDataFrame([(arr.tolist(),)], ["array"])\
            .withColumn("vec", arrayToVector(F.col("array")))\
            .select("vec").head().vec
        np.testing.assert_array_almost_equal(arr, vec.toArray())

    def test_vector_array(self):
        vec = Vectors.dense(1.0, 2.0, 3.0)
        arr = self.spark.createDataFrame([(vec,)], ["vec"])\
            .withColumn("arr", vectorToArray("vec"))\
            .select("arr").head().arr
        np.testing.assert_array_almost_equal(np.array(arr), vec.toArray())

    def test_vector_gen(self):
        row = self.spark.createDataFrame([(2, ), (3, )], ["size"])\
            .withColumn("vec_ones", onesVector("size"))\
            .withColumn("vec_zeros", zerosVector("size")).collect()
        np.testing.assert_array_almost_equal(row[0].vec_ones.toArray(), np.array([1, 1]))
        np.testing.assert_array_almost_equal(row[1].vec_ones.toArray(), np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(row[0].vec_zeros.toArray(), np.array([0, 0]))
        np.testing.assert_array_almost_equal(row[1].vec_zeros.toArray(), np.array([0, 0, 0]))

    def test_array_matrix(self):
        eye = np.array([1.0, 0.0, 0.0, 1.0])
        mat = self.spark.createDataFrame([(eye.tolist(),)], ["array"]) \
            .withColumn("mat", arrayToMatrix(2, 2, F.col("array"))) \
            .select("mat").head().mat
        np.testing.assert_array_almost_equal(eye.reshape(2, 2), mat.toArray())

    def test_matrix_array(self):
        mat = Matrices.dense(2, 2, [1.0, 2.0, 3.0, 4.0])
        arr = self.spark.createDataFrame([(mat,)], ["mat"]) \
            .withColumn("arr", matrixToArray("mat")) \
            .select("arr").head().arr
        np.testing.assert_array_almost_equal(np.array(arr.values), mat.toArray().reshape(4, order="F"))