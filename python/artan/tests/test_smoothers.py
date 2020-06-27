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
from artan.smoother import LinearKalmanSmoother
from pyspark.ml.linalg import Vectors, Matrices
import numpy as np


class LinearKalmanSmootherTests(ReusedSparkTestCase):

    np.random.seed(0)

    def test_ols_equivalence(self):
        # Simple ols problem
        # y =  a * x + b + r
        # Where r ~ N(0, 1)
        n = 40
        a = 0.4
        b = 0.5
        x = np.arange(0, n)
        r = np.random.normal(0, 1, n)
        y = (a * x + b + r).reshape(n, 1)
        features = x.reshape(n, 1)
        features = np.concatenate([features, np.ones_like(features)], axis=1)
        df = self.spark.createDataFrame(
            [(Vectors.dense(y[i]), Matrices.dense(1, 2, features[i])) for i in range(n)],
            ["measurement", "measurementModel"])
        lkf = LinearKalmanSmoother(2, 1)\
            .setMeasurementModelCol("measurementModel")\
            .setMeasurementCol("measurement")\
            .setInitialStateCovariance(Matrices.dense(2, 2, (np.eye(2)*10).reshape(4, 1)))\
            .setProcessModel(Matrices.dense(2, 2, np.eye(2).reshape(4, 1)))\
            .setProcessNoise(Matrices.dense(2, 2, np.zeros(4)))\
            .setMeasurementNoise(Matrices.dense(1, 1, [10E-5]))\
            .setFixedLag(n)

        model = lkf.transform(df)
        state = model.filter("stateIndex = {}".format(n)).collect()[0].state.mean.values

        # Check equivalence with least squares solution with numpy
        expected, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
        np.testing.assert_array_almost_equal(state, expected.reshape(2), decimal=5)