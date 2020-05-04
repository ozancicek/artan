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
from artan.mixture import MultivariateGaussianMixture
from pyspark.ml.linalg import Vectors
import numpy as np


class MultivariateGaussianMixtureTests(ReusedSparkTestCase):

    np.random.seed(0)
    weights = [0.2, 0.3, 0.5]
    means = [[10.0, 2.0], [1.0, 4.0], [5.0, 3.0]]
    covs = [[2.0, 1.0, 1.0, 2.0], [4.0, 0.0, 0.0, 4.0], [5.0, 3.0, 3.0, 5.0]]

    @staticmethod
    def generate_samples(weights, means, covs, size):
        samples = []
        for weight, mean, cov in zip(weights, means, covs):
            seq_size = int(weight * size)
            seq = np.random.multivariate_normal(
                np.array(mean), np.array(cov).reshape(len(mean), len(mean)), size=seq_size)
            samples.append(seq)

        samples_arr = np.concatenate(samples)
        np.random.shuffle(samples_arr)
        return [(Vectors.dense(sample),) for sample in samples_arr]

    @staticmethod
    def _mae(left, right):
        return np.mean(np.abs(left - right))

    def test_online_gmm(self):
        mb_size = 1
        sample_size = 5000

        samples = self.generate_samples(self.weights, self.means, self.covs, sample_size)

        samples_df = self.spark.createDataFrame(samples, ["sample"])

        eye = [1.0, 0.0, 0.0, 1.0]
        gmm = MultivariateGaussianMixture(3) \
            .setInitialMeans([[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]]) \
            .setInitialCovariances([eye, eye, eye]) \
            .setStepSize(0.01) \
            .setMinibatchSize(mb_size) \

        result = gmm.transform(samples_df) \
            .filter("stateIndex == {}".format(int(sample_size/mb_size))) \
            .collect()[0]

        mixture_model = result.mixtureModel
        mae_weights = self._mae(np.array(mixture_model.weights), np.array(self.weights))
        assert(mae_weights < 0.2)
        for i, dist in enumerate(mixture_model.distributions):
            mae_mean = self._mae(dist.mean.toArray(), np.array(self.means[i]))
            assert(mae_mean < 3)
            mae_cov = self._mae(
                dist.covariance.toArray().reshape(4), np.array(self.covs[i]))
            assert(mae_cov < 4)

    def test_minibatch_gmm(self):
        mb_size = 50
        sample_size = 5000

        samples = self.generate_samples(self.weights, self.means, self.covs, sample_size)

        samples_df = self.spark.createDataFrame(samples, ["sample"])

        eye = [1.0, 0.0, 0.0, 1.0]
        gmm = MultivariateGaussianMixture(3)\
            .setInitialMeans([[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]])\
            .setInitialCovariances([eye, eye, eye])\
            .setStepSize(0.6)\
            .setMinibatchSize(mb_size)

        result = gmm.transform(samples_df)\
            .filter("stateIndex == {}".format(int(sample_size/mb_size)))\
            .collect()[0]

        mixture_model = result.mixtureModel
        np.testing.assert_array_almost_equal(
            np.array(mixture_model.weights), np.array(self.weights), decimal=1)
        for i, dist in enumerate(mixture_model.distributions):
            mae_mean = self._mae(dist.mean.toArray(), np.array(self.means[i]))
            assert(mae_mean < 0.35)
            mae_cov = self._mae(
                dist.covariance.toArray().reshape(4), np.array(self.covs[i]))
            assert(mae_cov < 1.0)