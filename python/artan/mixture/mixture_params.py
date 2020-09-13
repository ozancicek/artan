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
        "sampleCol",
        "Column name for input to mixture models",
        TypeConverters.toString)

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
        "initialWeights", "Initial weights of the mixtures. The weights should sum up to 1.0 .",
        TypeConverters.toListFloat)

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
        "initialWeightsCol", "Initial weights of mixtures from dataframe column",
        TypeConverters.toString)

    def __init__(self):
        super(HasInitialWeightsCol, self).__init__()

    def getInitialWeightsCol(self):
        """
        Gets the value of initial weights or its default value.
        """
        return self.getOrDefault(self.initialWeightsCol)


class HasStepSize(Params):
    """
    Mixin for controlling the inertia of the current parameter.
    """

    stepSize = Param(
        Params._dummy(),
        "stepSize",
        "Weights the current parameter of the model against the old parameter. A step size of 1.0 means ignore" +
        "the old parameter, whereas a step size of 0 means ignore the current parameter. Values closer to 1.0 will" +
        "increase speed of convergence, but might have adverse effects on stability. In online setting," +
        "it is advised to set small values close to 0.0. Default is 0.01",
        TypeConverters.toFloat)

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
        "stepSizeCol",
        "stepSize parameter from dataframe column instead of a constant value across all samples",
        TypeConverters.toString)

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
        "decayRate",
        "Step size as a decaying function rather than a constant, which might be preferred in batch training." +
        "If set, the step size will be replaced with the output of the function" +
        "stepSize = (2 + kIter)**(-decayRate)", TypeConverters.toFloat)

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
        "minibatchSize",
        "Size for batching samples together in online EM algorithm. Estimate will be produced once per each batch" +
        "Having larger batches increases stability with increased memory footprint. Each minibatch is stored in" +
        "mixture transformer state independently from spark minibatches.",
        TypeConverters.toInt)

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
        "minibatchSizeCol",
        "Set minibatch size from dataframe column",
        TypeConverters.toString)

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
        "updateHoldout",
        "Controls after how many samples the mixture will start calculating estimates. Preventing update" +
        "in first few samples might be preferred for stability.",
        TypeConverters.toInt)

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
        "updateHoldoutCol",
        "updateHoldout from dataframe column rather than a constant value across all states",
        TypeConverters.toString)

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
        "Sets the initial mixture model from struct column conforming to mixture distribution",
        TypeConverters.toString)

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
        "Flag to enable batch EM. Unless enabled, the transformer will do online EM. Online EM can be done with" +
        "both streaming and batch dataframes, whereas batch EM can only be done with batch dataframes. Default is false",
        TypeConverters.toBoolean)

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
        "batchTrainMaxIter",
        "Maximum iterations in batch train mode, default is 30",
        TypeConverters.toInt)

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
        "batchTrainTol",
        "Min change in loglikelihood to stop iterations in batch EM mode. Default is 0.1",
        TypeConverters.toFloat)

    def __init__(self):
        super(HasBatchTrainTol, self).__init__()

    def getBatchTrainTol(self):
        """
        Gets the value of batchTrainTol or its default value
        """
        return self.getOrDefault(self.batchTrainTol)


class HasMixtureCount(Params):
    """
    Mixin for number of components in the mixture
    """
    mixtureCount = Param(
        Params._dummy(),
        "mixtureCount",
        "Number of finite mixture components, must ge > 0",
        TypeConverters.toInt
    )

    def __init__(self):
        super(HasMixtureCount, self).__init__()

    def getMixtureCount(self):
        """
        Gets the value of mixtureCount or its default value
        """
        return self.getOrDefault(self.mixtureCount)


class MixtureParams(HasSampleCol, HasStepSize, HasStepSizeCol, HasInitialWeights, HasInitialWeightsCol,
                    HasMinibatchSize, HasUpdateHoldout, HasDecayRate, HasInitialMixtureModelCol,
                    HasMinibatchSizeCol, HasUpdateHoldoutCol, HasBatchTrainEnabled, HasBatchTrainMaxIter,
                    HasBatchTrainTol, HasMixtureCount):

    def setMixtureCount(self, value):
        """
        Sets the number of components in the finite mixture

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(mixtureCount=value)

    def setSampleCol(self, value):
        """
        Sets the sample column for the mixture model inputs. Depending on the mixture distribution, sample type should
        be different.

        Bernoulli => Boolean
        Poisson => Long
        MultivariateGaussian => Vector

        :param value: String
        :return: MixtureTransformer
        """
        return self._set(sampleCol=value)

    def setStepSize(self, value):
        """
        Sets the step size parameter, which weights the current parameter of the model against the old parameter.
        A step size of 1.0 means ignore the old parameter, whereas a step size of 0 means ignore the current parameter.
        Values closer to 1.0 will increase speed of convergence, but might have adverse effects on stability. For online
        EM, it is advised to set it close to 0.0.

        Default is 0.1

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(stepSize=value)

    def setStepSizeCol(self, value):
        """
        Sets the step size from dataframe column, which would allow setting different step sizes accross measurements.
        Overrides the value set by setStepSize

        :param value: String
        :return: MixtureTransformer
        """
        return self._set(stepSizeCol=value)

    def setDecayRate(self, value):
        """
        Sets the step size as a decaying function rather than a constant step size, which might be preferred
        for batch training. If set, the step size will be replaced with the output of following function:

        stepSize = (2 + kIter)**(-decayRate)

        Where kIter is incremented by 1 on each minibatch.

        :return: MixtureTransformer
        """
        return self._set(decayRate=value)

    def setInitialWeights(self, value):
        """
        Sets the initial weights of the mixtures. The weights should sum up to 1.0.

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
        Sets the minibatch size for batching samples together in online EM algorithm. Estimate will be produced once
        per each batch. Having larger batches increases stability with increased memory footprint.

        Default is 1

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(minibatchSize=value)

    def setMinibatchSizeCol(self, value):
        """
        Sets the minibatch size from dataframe column rather than a constant minibatch size across all states.
        Overrides setMinibatchSize setting.

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(minibatchSizeCol=value)

    def setUpdateHoldout(self, value):
        """
        Sets the update holdout parameter which controls after how many samples the mixture will start calculating
        estimates. Preventing update in first few samples might be preferred for stability.

        :param value: Int
        :return: MixtureTransformer
        """
        return self._set(updateHoldout=value)

    def setUpdateHoldoutCol(self, value):
        """
        Sets the update holdout parameter from dataframe column rather than a constant value across all states.
        Overrides the value set by setUpdateHoldout

        :param value: String
        :return: MixtureTransormer
        """
        return self._set(updateHoldoutCol=value)

    def setInitialMixtureModelCol(self, value):
        """
        Sets the initial mixture model directly from dataframe column

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