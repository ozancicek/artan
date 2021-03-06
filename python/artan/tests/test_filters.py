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
from artan.filter import RecursiveLeastSquaresFilter, LinearKalmanFilter, LeastMeanSquaresFilter
from pyspark.ml.linalg import Vectors, Matrices
import numpy as np
import os
import tempfile


class RLSTests(ReusedSparkTestCase):

    np.random.seed(0)

    def test_simple_rls(self):
        df = self.spark.createDataFrame(
            [(1.0, Vectors.dense(0.0, 5.0)),
             (0.0, Vectors.dense(1.0, 2.0)),
             (1.0, Vectors.dense(2.0, 1.0)),
             (0.0, Vectors.dense(3.0, 3.0)), ], ["label", "features"])

        rls = RecursiveLeastSquaresFilter()\
            .setInitialEstimate(Vectors.dense(0.0, 0.0)) \
            .setFeatureSize(2) \
            .setRegularizationMatrixFactor(10E6)

        model = rls.transform(df).filter("stateIndex=4").collect()
        state = model[0].state.mean.values

        expected = np.array([5.31071176e-09, 1.53846148e-01])
        np.testing.assert_array_almost_equal(state, expected)

    def test_ols_equivalence(self):
        # Simple ols problem
        # y =  a * x + b + r
        # Where r ~ N(0, 1)
        n = 40
        a = 0.5
        b = 2
        x = np.arange(0, n)
        r = np.random.normal(0, 1, n)
        y = a * x + b + r
        features = x.reshape(n, 1)
        features = np.concatenate([features, np.ones_like(features)], axis=1)

        df = self.spark.createDataFrame(
            [(float(y[i]), Vectors.dense(features[i])) for i in range(n)], ["label", "features"])

        # set high regularization matrix factor to get close to OLS solution
        rls = RecursiveLeastSquaresFilter()\
            .setFeatureSize(2)\
            .setInitialEstimate(Vectors.dense([1.0, 1.0]))\
            .setRegularizationMatrixFactor(10E6)

        model = rls.transform(df)
        state = model.filter("stateIndex = {}".format(n)).collect()[0].state.mean.values

        # Check equivalence with least squares solution with numpy
        expected, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
        np.testing.assert_array_almost_equal(state, expected)

    def test_persistance(self):
        filter = RecursiveLeastSquaresFilter() \
            .setFeatureSize(3)\
            .setInitialEstimate(Vectors.dense([10.0, 0.0, 0.0]))

        path = tempfile.mkdtemp()
        model_path = os.path.join(path, "rls")
        filter.save(model_path)

        loaded = RecursiveLeastSquaresFilter.load(model_path)
        assert(loaded.getStateSize() == filter.getStateSize())
        assert(loaded.getInitialStateMean() == filter.getInitialStateMean())


class LMSTests(ReusedSparkTestCase):

    np.random.seed(0)

    def test_filter_trend(self):
        # y =  a * x + N(0, 1)
        n = 40
        a = 0.2
        x = np.arange(0, n)
        r = np.random.normal(0, 1, n)
        y = a * x + r
        features = x.reshape(n, 1)

        df = self.spark.createDataFrame(
            [(float(y[i]), Vectors.dense(features[i])) for i in range(n)], ["l", "f"])

        lms = LeastMeanSquaresFilter()\
            .setInitialEstimate(Vectors.dense([10.0]))\
            .setRegularizationConstant(1.0)\
            .setLearningRate(1.0)\
            .setLabelCol("l")\
            .setFeaturesCol("f")

        model = lms.transform(df)
        state = model.filter("stateIndex = {}".format(n)).collect()[0].state.values

        np.testing.assert_array_almost_equal(state, np.array([0.2]), 2)

    def test_persistance(self):
        filter = LeastMeanSquaresFilter() \
            .setFeaturesCol("some") \
            .setInitialEstimate(Vectors.dense([10.0])) \

        path = tempfile.mkdtemp()
        model_path = os.path.join(path, "lms")
        filter.save(model_path)

        loaded = LeastMeanSquaresFilter.load(model_path)
        assert(loaded.getFeaturesCol() == filter.getFeaturesCol())
        assert(loaded.getInitialStateMean() == filter.getInitialStateMean())


class LinearKalmanFilterTests(ReusedSparkTestCase):

    np.random.seed(0)

    def test_persistance(self):
        filter = LinearKalmanFilter() \
            .setStateSize(2) \
            .setInitialStateMean(Vectors.dense([0.0, 0.0])) \
            .setInitialStateCovariance(Matrices.dense(2, 2, [1.0, 0.0, 0.0, 0.0]))\

        path = tempfile.mkdtemp()
        model_path = os.path.join(path, "lkf")
        filter.save(model_path)

        loaded = LinearKalmanFilter.load(model_path)
        assert(loaded.getInitialStateMean() == filter.getInitialStateMean())
        assert(loaded.getInitialStateCovariance() == filter.getInitialStateCovariance())
        assert(loaded.getStateSize() == filter.getStateSize())

    def test_ols_equivalence(self):
        # Simple ols problem
        # y =  a * x + b + r
        # Where r ~ N(0, 1)
        n = 40
        a = 0.27
        b = 1.2
        x = np.arange(0, n)
        r = np.random.normal(0, 1, n)
        y = (a * x + b + r).reshape(n, 1)
        features = x.reshape(n, 1)
        features = np.concatenate([features, np.ones_like(features)], axis=1)
        df = self.spark.createDataFrame(
            [(Vectors.dense(y[i]), Matrices.dense(1, 2, features[i])) for i in range(n)],
            ["measurement", "measurementModel"])
        lkf = LinearKalmanFilter()\
            .setInitialStateMean(Vectors.dense(0.0, 0.0))\
            .setMeasurementModelCol("measurementModel")\
            .setMeasurementCol("measurement")\
            .setInitialStateCovariance(Matrices.dense(2, 2, (np.eye(2)*10).reshape(4, 1)))\
            .setProcessModel(Matrices.dense(2, 2, np.eye(2).reshape(4, 1)))\
            .setProcessNoise(Matrices.dense(2, 2, np.zeros(4)))\
            .setMeasurementNoise(Matrices.dense(1, 1, [10E-5]))

        model = lkf.transform(df)
        state = model.filter("stateIndex = {}".format(n)).collect()[0].state.mean.values

        # Check equivalence with least squares solution with numpy
        expected, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
        np.testing.assert_array_almost_equal(state, expected.reshape(2), decimal=5)

    def test_batch_save_and_resume(self):
        n = 100
        ts = np.arange(0, n)
        zs = np.random.normal(0, 1, n) + ts

        split_point = n//2
        initial = zs[:split_point]
        remaining = zs[split_point:]

        filter = LinearKalmanFilter()\
            .setMeasurementCol("measurement")\
            .setInitialStateMean(
                Vectors.dense([0.0, 0.0]))\
            .setInitialStateCovariance(
                Matrices.dense(2, 2, [1, 0, 0, 1]))\
            .setProcessModel(
                Matrices.dense(2, 2, [1, 0, 1, 1]))\
            .setProcessNoise(
                Matrices.dense(2, 2, [0.01, 0.0, 0.0, 0.01]))\
            .setMeasurementNoise(
                Matrices.dense(1, 1, [1]))\
            .setMeasurementModel(
                Matrices.dense(1, 2, [1, 0]))

        initial_filter = filter.setInitialStateCovariance(
                Matrices.dense(2, 2, [1000.0, 0.0, 0.0, 1000.0]))

        def create_df(m):
            return self.spark.createDataFrame(
                [(Vectors.dense(m[i]), ) for i in range(len(m))],
                ["measurement"])

        initial_measurements = create_df(initial)

        complete_measurements = create_df(zs)

        initial_state = initial_filter.transform(initial_measurements)\
            .filter("stateIndex == {}".format(len(initial)))\
            .select("stateKey", "state")

        complete_state = initial_filter.transform(complete_measurements) \
            .filter("stateIndex == {}".format(len(zs)))\
            .select("stateKey", "state")

        restarted_filter = filter\
            .setInitialStateDistributionCol("state")

        remaining_measurements = create_df(remaining)\
            .crossJoin(initial_state)

        restarted_state = restarted_filter.transform(remaining_measurements)\
            .filter("stateIndex == {}".format(n - split_point))\
            .select("stateKey", "state")

        assert(restarted_state.collect() == complete_state.collect())

    def test_multiple_model_adaptive_filter(self):
        n = 100
        a = 0.27
        b = 1.2
        x = np.concatenate([np.arange(0, n), np.arange(0, n)])
        r = np.random.normal(0, 1, n * 2)
        y = (a * x + b + r).reshape(n * 2, 1)
        features = x.reshape(n * 2, 1)
        features = np.concatenate([features, np.ones_like(features)], axis=1)
        state_keys = ["1"] * n + ["2"] * n
        df = self.spark.createDataFrame(
            [(state_keys[i], Vectors.dense(y[i]), Matrices.dense(1, 2, features[i])) for i in range(n*2)],
            ["state_key","measurement", "measurementModel"])

        mmaeFilter = LinearKalmanFilter() \
            .setStateSize(2) \
            .setInitialStateMean(Vectors.dense(0.0, 0.0)) \
            .setStateKeyCol("state_key")\
            .setMeasurementModelCol("measurementModel")\
            .setMeasurementCol("measurement")\
            .setInitialStateCovariance(Matrices.dense(2, 2, (np.eye(2)*10).reshape(4, 1)))\
            .setProcessModel(Matrices.dense(2, 2, np.eye(2).reshape(4, 1)))\
            .setProcessNoise(Matrices.dense(2, 2, np.zeros(4)))\
            .setMeasurementNoise(Matrices.dense(1, 1, [1.0]))\
            .setSlidingLikelihoodWindow(5)\
            .setEnableMultipleModelAdaptiveEstimation()

        model = mmaeFilter.transform(df)
        state = model.filter("stateIndex = {}".format(n)).collect()[0].state.mean.values

        expected, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
        np.testing.assert_array_almost_equal(state, expected.reshape(2), decimal=0)
