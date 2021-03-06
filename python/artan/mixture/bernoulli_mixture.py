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
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import JavaMLWritable
from artan.utils import ArtanJavaMLReadable

class _HasInitialProbabilities(Params):
    """
    Mixin for initial probabilities parameter.
    """

    initialProbabilities = Param(
        Params._dummy(),
        "initialProbabilities", "Initial Probabilities vectors of mixtures, as a list of floats")

    def __init__(self):
        super(_HasInitialProbabilities, self).__init__()

    def getInitialProbabilities(self):
        """
        Gets the value of initial probabilities or its default value.
        """
        return self.getOrDefault(self.initialProbabilities)


class _HasInitialProbabilitiesCol(Params):
    """
    Mixin for initial Probabilities parameter.
    """

    initialProbabilitiesCol = Param(
        Params._dummy(),
        "initialProbabilitiesCol",
        "Initial probabilities vectors of mixtures from dataframe column",
        TypeConverters.toString)

    def __init__(self):
        super(_HasInitialProbabilitiesCol, self).__init__()

    def getInitialProbabilitiesCol(self):
        """
        Gets the value of initial Probabilities or its default value.
        """
        return self.getOrDefault(self.initialProbabilitiesCol)


class _HasBernoulliMixtureModelCol(Params):
    """
    Mixin for Bernoulli mixture model parameter.
    """

    bernoulliMixtureModelCol = Param(
        Params._dummy(),
        "bernoulliMixtureModelCol",
        "Initial mixture model from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(_HasBernoulliMixtureModelCol, self).__init__()

    def getBernoulliMixtureModelCol(self):
        """
        Gets the value of cmm col or its default value.
        """
        return self.getOrDefault(self.bernoulliMixtureModelCol)


class BernoulliMixture(StatefulTransformer, MixtureParams, _HasInitialProbabilities, _HasInitialProbabilitiesCol,
                       _HasBernoulliMixtureModelCol, ArtanJavaMLReadable, JavaMLWritable):
    """
    Online bernoulli mixture estimator with a stateful transformer, based on Cappe (2011) Online
    Expectation-Maximisation paper.

    Outputs an estimate for each input sample in a single pass, by replacing the E-step in EM with a recursively
    averaged stochastic E-step.

    """

    def __init__(self):
        super(BernoulliMixture, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.github.ozancicek.artan.ml.mixture.BernoulliMixture", self.uid)

    def setInitialProbabilities(self, value):
        """
        Sets the initial bernoulli probabilities of the mixtures. The length of the array should be equal to mixture
        count, each element in the array should be between 0 and 1.

        Default is equally spaced probabilities between 0 and 1

        :param value: List[Float]
        :return: BernoulliMixture
        """
        return self._set(initialProbabilities=value)

    def setInitialProbabilitiesCol(self, value):
        """
        Sets the initial probabilities from dataframe column to set different probabilities across different models.
        Overrides the parameter set by setInitialProbabilities.

        :param value: String
        :return: BernoulliMixture
        """
        return self._set(initialProbabilitiesCol=value)

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        py_stage = BernoulliMixture()
        py_stage._java_obj = java_stage
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()
        return py_stage