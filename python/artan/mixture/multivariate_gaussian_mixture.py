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

from artan.state import StatefulTransformer
from artan.mixture.mixture_params import MixtureParams
from artan.utils import ArtanJavaMLReadable
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import JavaMLWritable


class _HasInitialMeans(Params):
    """
    Mixin for initial means parameter.
    """

    initialMeans = Param(
        Params._dummy(),
        "initialMeans", "Initial mean vectors of mixtures, as a list of list of floats")

    def __init__(self):
        super(_HasInitialMeans, self).__init__()

    def getInitialMeans(self):
        """
        Gets the value of initial means or its default value.
        """
        return self.getOrDefault(self.initialMeans)


class _HasInitialMeansCol(Params):
    """
    Mixin for initial means parameter.
    """

    initialMeansCol = Param(
        Params._dummy(),
        "initialMeansCol", "Initial mean vectors of mixtures from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(_HasInitialMeansCol, self).__init__()

    def getInitialMeansCol(self):
        """
        Gets the value of initial means or its default value.
        """
        return self.getOrDefault(self.initialMeansCol)


class _HasInitialCovariances(Params):
    """
    Mixin for initial means parameter.
    """

    initialCovariances = Param(
        Params._dummy(),
        "initialCovariances", "Initial covariance matrices of mixtures, as a list of list of floats")

    def __init__(self):
        super(_HasInitialCovariances, self).__init__()

    def getInitialCovariances(self):
        """
        Gets the value of initial covariances or its default value.
        """
        return self.getOrDefault(self.initialCovariances)


class _HasInitialCovariancesCol(Params):
    """
    Mixin for initial means parameter.
    """

    initialCovariancesCol = Param(
        Params._dummy(),
        "initialCovariancesCol",
        "Initial covariance matrices of mixtures from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(_HasInitialCovariancesCol, self).__init__()

    def getInitialCovariancesCol(self):
        """
        Gets the value of initial covariances or its default value.
        """
        return self.getOrDefault(self.initialCovariancesCol)


class MultivariateGaussianMixture(StatefulTransformer, MixtureParams, _HasInitialMeans, _HasInitialMeansCol,
                                  _HasInitialCovariances, _HasInitialCovariancesCol,
                                  ArtanJavaMLReadable, JavaMLWritable):
    """
    Online gaussian mixture estimator with a stateful transformer, based on Cappe (2011) Online
    Expectation-Maximisation paper.

    Outputs an estimate for each input sample in a single pass, by replacing the E-step in EM with a recursively
    averaged stochastic E-step.
    """

    def __init__(self):
        super(MultivariateGaussianMixture, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.github.ozancicek.artan.ml.mixture.MultivariateGaussianMixture", self.uid)

    def setInitialMeans(self, value):
        """
        Sets the initial mean vectors of the mixtures as a nested array of doubles. The dimensions of the array should
        be mixtureCount x sample vector size

        :param value: List[List[Float]]
        :return: MultivariateGaussianMixture
        """
        return self._set(initialMeans=value)

    def setInitialMeansCol(self, value):
        """
        Sets the initial means from dataframe column. Overrides the value set by setInitialMeans.

        :param value: String
        :return: MultivariateGaussianMixture
        """
        return self._set(initialMeansCol=value)

    def setInitialCovariances(self, value):
        """
        Sets the initial covariance matrices of the mixtures as a nested array of doubles. The dimensions of the array
        should be mixtureCount x sampleSize**2

        :param value: List[List[Float]]
        :return: MultivariateGaussianMixture
        """
        return self._set(initialCovariances=value)

    def setInitialCovariancesCol(self, value):
        """
        Sets the initial covariance matrices of the mixtures from dataframe column. Overrides the value set
        by setInitialCovariances

        :param value: String
        :return: MultivariateGaussianMixture
        """
        return self._set(initialCovariancesCol=value)

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        py_stage = MultivariateGaussianMixture()
        py_stage._java_obj = java_stage
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()
        return py_stage