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

import numpy as np

from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.linalg import DenseMatrix
from artan.state import StatefulTransformer
from artan.filter.filter_params import HasInitialState


class HasForgettingFactor(Params):
    """
    Mixin for param forgetting factor, between 0 and 1.
    """

    forgettingFactor = Param(
        Params._dummy(),
        "forgettingFactor", "Forgetting factor between 0 and 1", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasForgettingFactor, self).__init__()

    def getForgettingFactor(self):
        """
        Gets the value of forgettingFactor or its default value.
        """
        return self.getOrDefault(self.forgettingFactor)


class HasRegularizationMatrix(Params):
    """
    Mixin for param regularization matrix.
    """

    regularizationMatrix = Param(
        Params._dummy(),
        "regularizationMatrix",
        "Positive definite regularization matrix for RLS filter, typically a factor multiplied by identity matrix." +
        "Small factors (>1) give more weight to initial state, whereas large factors (>>1) decrease regularization" +
        "and cause RLS filter to behave like ordinary least squares",
        typeConverter=TypeConverters.toMatrix)

    def __init__(self):
        super(HasRegularizationMatrix, self).__init__()

    def getRegularizationMatrix(self):
        """
        Gets the value of regularization matrix or its default value.
        """
        return self.getOrDefault(self.regularizationMatrix)


class RecursiveLeastSquaresFilter(StatefulTransformer, HasInitialState,
                                  HasForgettingFactor, HasRegularizationMatrix):
    """
    Recursive formulation of least squares with exponential weighting & regularization, implemented with
    a stateful spark Transformer for running parallel filters /w spark dataframes. Transforms an input dataframe
    of observations to a dataframe of model parameters using stateful spark transormations, which can be used
    in both streaming and batch applications.

    Let w denote the model parameters and w_est denote our prior belief. RLS minimizes following regularization
    & weighted SSE terms;

    (w - w_est).T * (lambda^(-N-1) * P)^(-1) * (w - w_est) + Sum(lambda(N - j)*(d_k - u_k.T * w),  k = 0,1, .. N)

    Where:
    - lambda: forgetting factor, or exponential weighting factor. Between 0 and 1.
    - P: regularization matrix. Smaller values increseas the weight of regularization term, whereas larger values
      increases the weight of weighted SSE term.
    - d_k, u_k: label and features vector at time step k.
    """
    def __init__(self, featuresSize):
        super(RecursiveLeastSquaresFilter, self).__init__()
        self._java_obj = self._new_java_obj("com.ozancicek.artan.ml.filter.RecursiveLeastSquaresFilter",
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

        :param value: pyspark.ml.linalg.Vector with size (featuresSize)
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(initialState=value)

    def setForgettingFactor(self, value):
        """
        Set forgetting factor, exponentially weights the measurements based on its sequence.

        Default value of 1.0 weights all measurements equally. With smaller values, recent measurements will have
        more weight. Generally set around 0.95 ~ 0.99

        :param value: Float >= 1.0
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(forgettingFactor=value)

    def setRegularizationMatrix(self, value):
        """
        Set regularization matrix governing the influence of the initial estimate (prior). Larger values will
        remove regularization effect, making the filter behave like OLS.

        Default is 10E5 * I

        :param value: pyspark.ml.linalg.Matrix with size (featuresSize, featuresSize)
        :return: RecursiveLeastSquaresFilter
        """
        return self._set(regularizationMatrix=value)

    def setRegularizationMatrixFactor(self, value):
        """
        Sets the regularization matrix with a float factor, which results in setting the regularization matrix as
        factor * identity

        :param value: Float
        :return: RecursiveLeastSquaresFilter
        """
        regMat = np.eye(self._featuresSize) * value
        rows, cols = regMat.shape
        return self._set(regularizationMatrix=DenseMatrix(rows, cols, regMat.reshape(rows * cols, order="F")))