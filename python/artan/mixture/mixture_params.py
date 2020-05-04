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


class HasSampleCol(Params):
    """
    Mixin for sample column parameter.
    """

    sampleCol = Param(
        Params._dummy(),
        "sampleCol", "Input sample column", TypeConverters.toString)

    def __init__(self):
        super(HasSampleCol, self).__init__()

    def getSampleCol(self):
        """
        Gets the value of initial weights or its default value.
        """
        return self.getOrDefault(self.sampleCol)


class HasInitialWeights(Params):
    """
    Mixin for initial mixture weights parameter.
    """

    initialWeights = Param(
        Params._dummy(),
        "initialWeights", "Initial weights of mixtures", TypeConverters.toListFloat)

    def __init__(self):
        super(HasInitialWeights, self).__init__()

    def getInitialWeights(self):
        """
        Gets the value of initial weights or its default value.
        """
        return self.getOrDefault(self.initialWeights)


class HasInitialWeightsCol(Params):
    """
    Mixin for initial mixture weights parameter.
    """

    initialWeightsCol = Param(
        Params._dummy(),
        "initialWeightsCol", "Initial weights of mixtures from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(HasInitialWeightsCol, self).__init__()

    def getInitialWeightsCol(self):
        """
        Gets the value of initial weights or its default value.
        """
        return self.getOrDefault(self.initialWeightsCol)


class HasStepSize(Params):
    """
    Mixin for step size parameter
    """

    stepSize = Param(
        Params._dummy(),
        "stepSize", "Step size for stochastic expectation approximation", TypeConverters.toFloat)

    def __init__(self):
        super(HasStepSize, self).__init__()

    def getStepSize(self):
        """
        Gets the value of step size or its default value
        """
        return self.getOrDefault(self.stepSize)


class HasStepSizeCol(Params):
    """
    Mixin for step size parameter
    """

    stepSizeCol = Param(
        Params._dummy(),
        "stepSizeCol", "Step size parameter from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(HasStepSizeCol, self).__init__()

    def getStepSizeCol(self):
        """
        Gets the value of step size or its default value
        """
        return self.getOrDefault(self.getStepSizeCol)


class HasDecayingStepSizeEnabled(Params):
    """
    Mixin for decaying step size parameter
    """
    decayingStepSizeEnabled = Param(
        Params._dummy(),
        "decayingStepSizeEnabled", "Param for enabling decaying step size", TypeConverters.toBoolean)

    def __init__(self):
        super(HasDecayingStepSizeEnabled, self).__init__()

    def getDecayingStepSizeEnabled(self):
        """
        Gets the value of decaying step size flag
        """
        return self.getOrDefault(self.decayingStepSizeEnabled)


class HasMinibatchSize(Params):
    """
    Mixin for mini-batch size parameter
    """

    minibatchSize = Param(
        Params._dummy(),
        "minibatchSize", "Mini-batch size controlling the number of samples in a batch", TypeConverters.toInt)

    def __init__(self):
        super(HasMinibatchSize, self).__init__()

    def getMinibatchSize(self):
        """
        Gets the value of step size or its default value
        """
        return self.getOrDefault(self.minibatchSize)


class HasUpdateHoldout(Params):
    """
    Mixin for update holdout parameter
    """

    updateHoldout = Param(
        Params._dummy(),
        "updateHoldout", "Update holdout for preventing updates in initial iterations", TypeConverters.toInt)

    def __init__(self):
        super(HasUpdateHoldout, self).__init__()

    def getUpdateHoldout(self):
        """
        Gets the value of step size or its default value
        """
        return self.getOrDefault(self.updateHoldout)


class MixtureParams(HasSampleCol, HasStepSize, HasStepSizeCol, HasInitialWeights, HasInitialWeightsCol,
                    HasMinibatchSize, HasUpdateHoldout, HasDecayingStepSizeEnabled):

    def setSampleCol(self, value):
        """
        Sets the sample column parameter

        :param value: String
        :return: MixtureTransformer
        """
        return self._set(sampleCol=value)

    def setStepSize(self, value):
        """
        Sets the step size parameter

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(stepSize=value)

    def setStepSizeCol(self, value):
        """
        Sets the step size column parameter

        :param value: String
        :return: MixtureTransformer
        """
        return self._set(stepSizeCol=value)

    def setEnableDecayingStepSize(self):
        """
        Enables decaying step size
        :return: MixtureTransformer
        """
        return self._set(decayingStepSizeEnabled=True)

    def setInitialWeights(self, value):
        """
        Sets the initial mixture weights parameter

        :param value: List[Float]
        :return: MixtureTransformer
        """
        return self._set(initialWeights=value)

    def setInitialWeightsCol(self, value):
        """
        Sets the initial mixture weights parameter from dataframe column

        :param value: String
        :return: MixtureTransformer
        """
        return self._set(initialWeightsCol=value)

    def setMinibatchSize(self, value):
        """
        Sets the mini-batch size parameter

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(minibatchSize=value)

    def setUpdateHoldout(self, value):
        """
        Sets the update holdout parameter

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(updateHoldout=value)