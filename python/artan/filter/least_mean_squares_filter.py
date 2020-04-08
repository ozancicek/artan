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

from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.common import inherit_doc
from pyspark.ml.param.shared import HasLabelCol, HasFeaturesCol
from artan.state import StatefulTransformer
from artan.filter.filter_params import HasInitialState


class _HasLearningRate(Params):
    """
    Mixin for param Normalized LMS learning rate
    """

    learningRate = Param(
        Params._dummy(),
        "learningRate",
        "Learning rate for Normalized LMS. If there is no interference, the default value of 1.0 is optimal",
        typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(_HasLearningRate, self).__init__()

    def getLearningRate(self):
        """
        Gets the value of learning rate or its default value.
        """
        return self.getOrDefault(self.learningRate)


class _HasRegularizationConstant(Params):
    """
    Mixin for param for regularization constant.
    """

    regularizationConstant = Param(
        Params._dummy(),
        "regularizationConstant",
        "Regularization term for stability, default is 1.0",
        typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(_HasRegularizationConstant, self).__init__()

    def getRegularizationConstant(self):
        """
        Gets the value of regularization constant or its default value.
        """
        return self.getOrDefault(self.regularizationConstant)


@inherit_doc
class LeastMeanSquaresFilter(StatefulTransformer, HasInitialState,
                             _HasLearningRate, _HasRegularizationConstant,
                             HasLabelCol, HasFeaturesCol):
    """
    Normalized Least Mean Squares filter, implemented with a stateful spark Transformer for running parallel
    filters /w spark dataframes. Transforms an input dataframe of observations to a dataframe of model parameters
    using stateful spark transformations, which can be used in both streaming and batch applications.

    Belonging to stochastic gradient descent type of methods, LMS minimizes SSE on each measurement
    based on the expectation of steepest descending gradient.

    Let :math:`w` denote the model parameter vector, :math:`u` denote the features vector,
    and :math:`d` the label corresponding to :math:`u`. Normalized LMS computes :math:`w` at step :math:`k`
    recursively by:

    .. math ::

        e &= d - u \cdot w_{k-1} \\

        w_k &= w_{k-1} + m * e * u * (c + u \cdot u)^{-1}


    Where
    - :math:`m`: Learning rate
    - :math:`c`: Regularization constant
    """
    def __init__(self, featuresSize):
        super(LeastMeanSquaresFilter, self).__init__()
        self._java_obj = self._new_java_obj("com.github.ozancicek.artan.ml.filter.LeastMeanSquaresFilter",
                                            featuresSize, self.uid)
        self._featuresSize = featuresSize

    def setLabelCol(self, value):
        """
        Set label column. Default is "label"

        :param value: String
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(labelCol=value)

    def setFeaturesCol(self, value):
        """
        Set features column. Default is "features"

        :param value: String
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(featuresCol=value)

    def setInitialEstimate(self, value):
        """
        Set initial estimate for model parameters. Default is zero vector.

        Note that if this parameter is set through here, it will result in same initial estimate for all filters.
        For different initial estimates across filters, set the dataframe column for corresponding to initial estimate
        with setInitialEstimateCol.

        :param value: pyspark.ml.linalg.Vector with size (featuresSize)
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(initialState=value)

    def setInitialEstimateCol(self, value):
        """
        Sets the column corresponding to initial estimates
        :param value: String
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(initialStateCol=value)

    def setLearningRate(self, value):
        """
        Set learning rate controlling the speed of convergence. Without noise, 1.0 is optimal.

        Default is 1.0

        :param value: Float
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(learningRate=value)

    def setRegularizationConstant(self, value):
        """
        Set constant for regularization controlling stability. Larger values increase stability but degrade
        convergence performance. Generally set to a small constant.

        Default is 1.0

        :param value: Float
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(regularizationConstant=value)