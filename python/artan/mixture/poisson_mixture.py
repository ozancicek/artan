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


class _HasInitialRates(Params):
    """
    Mixin for initial poisson rates parameter.
    """

    initialRates = Param(
        Params._dummy(),
        "initialRates", "Initial poisson rates of mixtures, as a list of floats", TypeConverters.toListFloat)

    def __init__(self):
        super(_HasInitialRates, self).__init__()

    def getInitialRates(self):
        """
        Gets the value of initial rates or its default value.
        """
        return self.getOrDefault(self.initialRates)


class _HasInitialRatesCol(Params):
    """
    Mixin for initial poisson rates parameter.
    """

    initialRatesCol = Param(
        Params._dummy(),
        "initialRatesCol", "Initial poisson rates of mixtures from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(_HasInitialRatesCol, self).__init__()

    def getInitialRatesCol(self):
        """
        Gets the value of initial rates column or its default value.
        """
        return self.getOrDefault(self.initialRatesCol)


class PoissonMixture(StatefulTransformer, MixtureParams, _HasInitialRates, _HasInitialRatesCol):
    """
    Online poisson mixture estimator with a stateful transformer, based on Cappe (2011) Online
    Expectation-Maximisation paper.

    Outputs an estimate for each input sample in a single pass, by replacing the E-step in EM with a recursively
    averaged stochastic E-step.
    """

    def __init__(self, mixtureCount):
        super(PoissonMixture, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.github.ozancicek.artan.ml.mixture.PoissonMixture", mixtureCount, self.uid)

    def setInitialRates(self, value):
        """
        Sets the initial poisson rates of the mixtures. The length of the array should be equal to mixtureCount.

        :param value: List[Float]
        :return: PoissonMixture
        """
        return self._set(initialRates=value)

    def setInitialRatesCol(self, value):
        """
        Sets the initial rates from dataframe column. Overrides the parameter set from setInitialRates.

        :param value: String
        :return: PoissonMixture
        """
        return self._set(initialRatesCol=value)