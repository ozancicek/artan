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


class HasDecayRate(Params):
    """
    Mixin for decaying step size parameter
    """
    decayRate = Param(
        Params._dummy(),
        "decayRate", "Param for enabling decaying step size", TypeConverters.toFloat)

    def __init__(self):
        super(HasDecayRate, self).__init__()

    def getDecayingStepSizeEnabled(self):
        """
        Gets the value of decaying step size flag
        """
        return self.getOrDefault(self.decayRate)


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
        Gets the value of minibatch size or its default value
        """
        return self.getOrDefault(self.minibatchSize)


class HasMinibatchSizeCol(Params):
    """
    Mixin for mini-batch size parameter
    """

    minibatchSizeCol = Param(
        Params._dummy(),
        "minibatchSizeCol", "Set minibatch size from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(HasMinibatchSizeCol, self).__init__()

    def getMinibatchSizeCol(self):
        """
        Gets the value of minibatch size column or its default value
        """
        return self.getOrDefault(self.minibatchSizeCol)


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
        Gets the value of update holdout or its default value
        """
        return self.getOrDefault(self.updateHoldout)


class HasUpdateHoldoutCol(Params):
    """
    Mixin for update holdout parameter
    """

    updateHoldoutCol = Param(
        Params._dummy(),
        "updateHoldoutCol", "Update holdout parameter from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(HasUpdateHoldoutCol, self).__init__()

    def getUpdateHoldoutCol(self):
        """
        Gets the value of update holdout col or its default value
        """
        return self.getOrDefault(self.updateHoldoutCol)


class HasInitialMixtureModelCol(Params):
    """
    Mixin for initial mixture model parameter.
    """

    initialMixtureModelCol = Param(
        Params._dummy(),
        "initialMixtureModelCol",
        "Initial mixture model from dataframe column", TypeConverters.toString)

    def __init__(self):
        super(HasInitialMixtureModelCol, self).__init__()

    def getInitialMixtureModelCol(self):
        """
        Gets the value of initial mixture model col or its default value.
        """
        return self.getOrDefault(self.initialMixtureModelCol)


class HasBatchTrainEnabled(Params):
    """
    Mixin for enabling batch EM train mode
    """

    batchTrainEnabled = Param(
        Params._dummy(),
        "batchTrainEnabled",
        "Flag for enabling batch train mode", TypeConverters.toBoolean)

    def __init__(self):
        super(HasBatchTrainEnabled, self).__init__()

    def getBatchTrainEnabled(self):
        """
        Gets the value of batch train flag or its default value
        """
        return self.getOrDefault(self.batchTrainEnabled)


class HasBatchTrainMaxIter(Params):
    """
    Mixin for batch train max iterations
    """

    batchTrainMaxIter = Param(
        Params._dummy(),
        "batchTrainMaxIter", "Max number of iterations in batch EM train mode", TypeConverters.toInt)

    def __init__(self):
        super(HasBatchTrainMaxIter, self).__init__()

    def getBatchTrainMaxIter(self):
        """
        Gets the value of maxIter or its default value
        """
        return self.getOrDefault(self.batchTrainMaxIter)


class HasBatchTrainTol(Params):
    """
    Mixin for batch train iteration stop tolerance
    """

    batchTrainTol = Param(
        Params._dummy(),
        "batchTrainTol", "Min change in loglikelihood to stop iterations in batch EM mode", TypeConverters.toFloat)

    def __init__(self):
        super(HasBatchTrainTol, self).__init__()

    def getBatchTrainTol(self):
        """
        Gets the value of batchTrainTol or its default value
        """
        return self.getOrDefault(self.batchTrainTol)


class MixtureParams(HasSampleCol, HasStepSize, HasStepSizeCol, HasInitialWeights, HasInitialWeightsCol,
                    HasMinibatchSize, HasUpdateHoldout, HasDecayRate, HasInitialMixtureModelCol,
                    HasMinibatchSizeCol, HasUpdateHoldoutCol, HasBatchTrainEnabled, HasBatchTrainMaxIter,
                    HasBatchTrainTol):

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

    def setDecayRate(self, value):
        """
        Enables decaying step size
        :return: MixtureTransformer
        """
        return self._set(decayRate=value)

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
        Sets the minibatch size parameter

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(minibatchSize=value)

    def setMinibatchSizeCol(self, value):
        """
        Sets the minibatch size parameter from dataframe column

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(minibatchSizeCol=value)

    def setUpdateHoldout(self, value):
        """
        Sets the update holdout parameter

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(updateHoldout=value)

    def setUpdateHoldoutCol(self, value):
        """
        Sets the update holdout parameter from dataframe column

        :param value: String
        :return: MixtureTransormer
        """
        return self._set(updateHoldoutCol=value)

    def setInitialMixtureModelCol(self, value):
        """
        Sets the initial mixture model column

        :param value: String
        :return: MixtureTransformer
        """
        return self._set(initialMixtureModel=value)

    def setEnableBatchTrain(self):
        """
        Enables batch train mode.

        :return: MixtureTransformer
        """
        return self._set(batchTrainEnabled=True)

    def setBatchTrainMaxIter(self, value):
        """
        Sets the max number of iterations in batch train mode

        Default is 30

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(batchTrainMaxIter=value)

    def setBatchTrainTol(self, value):
        """
        Sets the minimum loglikelihood improvement for stopping iterations in batch EM train mode

        Defaullt is 0.1

        :param value: Float
        :return: MixtureTransformer
        """
        return self._set(batchTrainTol=value)