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


class HasInitialState(Params):
    """
    Mixin for initial state vector.
    """

    initialState = Param(
        Params._dummy(),
        "initialState", "Initial state vector", typeConverter=TypeConverters.toVector)

    def __init__(self):
        super(HasInitialState, self).__init__()

    def getInitialState(self):
        """
        Gets the value of initial state vector or its default value.
        """
        return self.getOrDefault(self.initialState)


class HasInitialCovariance(Params):
    """
    Mixin for param initial covariance matrix.
    """

    initialCovariance = Param(
        Params._dummy(),
        "initialCovariance",
        "Initial covariance matrix",
        typeConverter=TypeConverters.toMatrix)

    def __init__(self):
        super(HasInitialCovariance, self).__init__()

    def getInitialCovariance(self):
        """
        Gets the value of initial covariance matrix or its default value.
        """
        return self.getOrDefault(self.initialCovariance)


class HasProcessModel(Params):
    """
    Mixin for param process model matrix.
    """

    processModel = Param(
        Params._dummy(),
        "processModel",
        "Process model matrix, transitions the state to the next state when applied",
        typeConverter=TypeConverters.toMatrix)

    def __init__(self):
        super(HasProcessModel, self).__init__()

    def getProcessModel(self):
        """
        Gets the value of process model matrix or its default value.
        """
        return self.getOrDefault(self.processModel)


class HasFadingFactor(Params):
    """
    Mixin for param fading factor.
    """

    fadingFactor = Param(
        Params._dummy(),
        "fadingFactor",
        "Factor controlling the weight of older measurements. With larger factor, more weights will" +
        "be given to recent measurements. Typically, should be really close to 1",
        typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasFadingFactor, self).__init__()

    def getFadingFactor(self):
        """
        Gets the value of fading factor or its default value.
        """
        return self.getOrDefault(self.fadingFactor)


class HasMeasurementModel(Params):
    """
    Mixin for param measurement model matrix.
    """

    measurementModel = Param(
        Params._dummy(),
        "measurementModel",
        "Measurement matrix, when multiplied with the state it should give the measurement vector",
        typeConverter=TypeConverters.toMatrix)

    def __init__(self):
        super(HasMeasurementModel, self).__init__()

    def getMeasurementModel(self):
        """
        Gets the value of measurement model matrix or its default value.
        """
        return self.getOrDefault(self.measurementModel)


class HasProcessNoise(Params):
    """
    Mixin for param process noise matrix.
    """

    processNoise = Param(
        Params._dummy(),
        "processNoise",
        "Process noise matrix",
        typeConverter=TypeConverters.toMatrix)

    def __init__(self):
        super(HasProcessNoise, self).__init__()

    def getProcessNoise(self):
        """
        Gets the value of process noise matrix or its default value.
        """
        return self.getOrDefault(self.processNoise)


class HasMeasurementNoise(Params):
    """
    Mixin for param measurement noise matrix.
    """

    measurementNoise = Param(
        Params._dummy(),
        "measurementNoise",
        "Measurement noise matrix",
        typeConverter=TypeConverters.toMatrix)

    def __init__(self):
        super(HasMeasurementNoise, self).__init__()

    def getMeasurementNoise(self):
        """
        Gets the value of measurement noise matrix or its default value.
        """
        return self.getOrDefault(self.measurementNoise)


class HasMeasurementCol(Params):
    """
    Mixin for param for measurement column.
    """

    measurementCol = Param(
        Params._dummy(),
        "measurementCol",
        "Column name for measurement vector. Missing measurements are allowed with nulls in the data",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasMeasurementCol, self).__init__()

    def getMeasurementCol(self):
        """
        Gets the value of measurement column or its default value.
        """
        return self.getOrDefault(self.measurementCol)


class HasMeasurementModelCol(Params):
    """
    Mixin for param for measurement model column.
    """

    measurementModelCol = Param(
        Params._dummy(),
        "measurementModelCol",
        "Column name for specifying measurement model from input DataFrame rather than" +
        "a constant measurement model for all filters",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasMeasurementModelCol, self).__init__()

    def getMeasurementModelCol(self):
        """
        Gets the value of measurement model column or its default value.
        """
        return self.getOrDefault(self.measurementModelCol)


class HasMeasurementNoiseCol(Params):
    """
    Mixin for param for measurement noise column.
    """

    measurementNoiseCol = Param(
        Params._dummy(),
        "measurementNoiseCol",
        "Column name for specifying measurement noise from input DataFrame rather than" +
        "a constant measurement noise for all filters",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasMeasurementNoiseCol, self).__init__()

    def getMeasurementNoiseCol(self):
        """
        Gets the value of measurement noise column or its default value.
        """
        return self.getOrDefault(self.measurementNoiseCol)


class HasProcessModelCol(Params):
    """
    Mixin for param for process model column.
    """

    processModelCol = Param(
        Params._dummy(),
        "processModelCol",
        "Column name for specifying process model from input DataFrame rather than" +
        "a constant measurement model for all filters",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasProcessModelCol, self).__init__()

    def getProcessModelCol(self):
        """
        Gets the value of process model column or its default value.
        """
        return self.getOrDefault(self.processModelCol)


class HasProcessNoiseCol(Params):
    """
    Mixin for param for process noise column.
    """

    processNoiseCol = Param(
        Params._dummy(),
        "processNoiseCol",
        "Column name for specifying process noise from input DataFrame rather than" +
        "a constant measurement noise for all filters",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasProcessNoiseCol, self).__init__()

    def getProcessNoiseCol(self):
        """
        Gets the value of process noise column or its default value.
        """
        return self.getOrDefault(self.processNoiseCol)


class HasControlCol(Params):
    """
    Mixin for param for control column.
    """

    controlCol = Param(
        Params._dummy(),
        "controlCol",
        "Column name for specifying control vector",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasControlCol, self).__init__()

    def getControlCol(self):
        """
        Gets the value of control column or its default value.
        """
        return self.getOrDefault(self.controlCol)


class HasControlFunctionCol(Params):
    """
    Mixin for param for control function column.
    """

    controlFunctionCol = Param(
        Params._dummy(),
        "controlFunctionCol",
        "Column name for specifying control matrix",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasControlFunctionCol, self).__init__()

    def getControlFunctionCol(self):
        """
        Gets the value of control function column or its default value.
        """
        return self.getOrDefault(self.controlFunctionCol)


class HasCalculateMahalanobis(Params):
    """
    Mixin for param for enabling mahalanobis calculation.
    """

    calculateMahalanobis = Param(
        Params._dummy(),
        "calculateMahalanobis",
        "When true, mahalanobis distance of residual will be calculated & added to output DataFrame." +
        "Default is false.",
        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasCalculateMahalanobis, self).__init__()

    def getCalculateMahalanobis(self):
        """
        Gets the value of mahalanobis calcuation flag.
        """
        return self.getOrDefault(self.calculateMahalanobis)


class HasCalculateLoglikelihood(Params):
    """
    Mixin for param for enabling loglikelihood calculation.
    """

    calculateLoglikelihood= Param(
        Params._dummy(),
        "calculateLoglikelihood",
        "When true, loglikelihood of residual will be calculated & added to output DataFrame. Default is false",
        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasCalculateLoglikelihood, self).__init__()

    def getCalculateLoglikelihood(self):
        """
        Gets the value of loglikelihood calculation flag.
        """
        return self.getOrDefault(self.calculateLoglikelihood)