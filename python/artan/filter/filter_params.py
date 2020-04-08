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


class HasInitialStateCol(Params):
    """
    Mixin for param for initial state column.
    """

    initialStateCol = Param(
        Params._dummy(),
        "initialStateCol",
        "Column name for initial state vector.",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasInitialStateCol, self).__init__()

    def getInitialStateCol(self):
        """
        Gets the value of initial state column or its default value.
        """
        return self.getOrDefault(self.initialStateCol)


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


class HasInitialCovarianceCol(Params):
    """
    Mixin for param for initial covariance column.
    """

    initialCovarianceCol = Param(
        Params._dummy(),
        "initialCovarianceCol",
        "Column name for initial covariance vector.",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasInitialCovarianceCol, self).__init__()

    def getInitialCovarianceCol(self):
        """
        Gets the value of initial covariance column or its default value.
        """
        return self.getOrDefault(self.initialCovarianceCol)


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

    calculateLoglikelihood = Param(
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


class HasOutputSystemMatrices(Params):
    """
    Mixin for param for enabling the output of system matrices along with the state.
    """

    outputSystemMatrices = Param(
        Params._dummy(),
        "outputSystemMatrices",
        "When true, the system matrices will be added to output DataFrame. Default is false",
        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasOutputSystemMatrices, self).__init__()

    def getOutputSystemMatrices(self):
        """
        Gets the value of loglikelihood calculation flag.
        """
        return self.getOrDefault(self.outputSystemMatrices)


class HasCalculateSlidingLikelihood(Params):
    """
    Mixin param for enabling sliding likelihood calculation
    """
    calculateSlidingLikelihood = Param(
        Params._dummy(),
        "calculateSlidingLikelihood",
        "When true, sliding likelihood sum of residual will be calculated & added to output DataFrame. Default is false",
        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasCalculateSlidingLikelihood, self).__init__()

    def getCalculateSlidingLikelihood(self):
        """
        Gets the value of sliding likelihood calculation flag
        """
        return self.getOrDefault(self.calculateSlidingLikelihood)


class HasSlidingLikelihoodWindow(Params):
    """
    Mixin param for sliding likelihood window duration
    """
    slidingLikelihoodWindow = Param(
        Params._dummy(),
        "slidingLikelihoodWindow",
        "Number of consecutive measurements to include in the total likelihood calculation",
        typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasSlidingLikelihoodWindow, self).__init__()

    def getSlidingLikelihoodWindow(self):
        """
        Gets the value of sliding likelihood window
        """
        return self.getOrDefault(self.slidingLikelihoodWindow)


class HasMultipleModelMeasurementWindowDuration(Params):

    multipleModelMeasurementWindowDuration = Param(
        Params._dummy(),
        "multipleModelMeasurementWindowDuration",
        "Window duration for grouping measurements in same window for MMAE filter aggregation",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasMultipleModelMeasurementWindowDuration, self).__init__()

    def getMultipleModelMeasurementWindowDuration(self):
        """
        Gets the value of mmae measureent window duration
        """
        return self.getOrDefault(self.multipleModelMeasurementWindowDuration)


class HasMultipleModelAdaptiveEstimationEnabled(Params):

    multipleModelAdaptiveEstimationEnabled = Param(
        Params._dummy(),
        "multipleModelAdaptiveEstimationEnabled",
        "Flag for enabling  Multiple Model Adaptive Estimation (MMAE) output mode. When enabled," + "" +
        "MMAE mode outputs a single state estimate from the output of all kalman states of the transformer." +
        "States are weighted based on their sliding likelihood",
        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasMultipleModelAdaptiveEstimationEnabled, self).__init__()

    def getMultipleModelAdaptiveEstimationEnabled(self):
        """
        Gets the value of MMAE output mode flag
        """
        return self.getOrDefault(self.multipleModelAdaptiveEstimationEnabled)


class KalmanFilterParams(HasInitialState, HasInitialCovariance, HasInitialStateCol,
                         HasInitialCovarianceCol, HasProcessModel, HasFadingFactor, HasMeasurementModel,
                         HasMeasurementNoise, HasProcessNoise, HasMeasurementCol,
                         HasMeasurementModelCol, HasMeasurementNoiseCol, HasProcessModelCol,
                         HasProcessNoiseCol, HasControlCol, HasControlFunctionCol,
                         HasCalculateMahalanobis, HasCalculateLoglikelihood,
                         HasOutputSystemMatrices, HasCalculateSlidingLikelihood,
                         HasSlidingLikelihoodWindow):
    """
    Mixin for kalman filter parameters
    """
    def setInitialState(self, value):
        """
        Set the initial state vector with size (stateSize).

        It will be applied to all states. If the state timeouts and starts receiving
        measurements after timeout, it will again start from this initial state vector. Default is zero.

        Note that if this parameter is set through here, it will result in same initial state for all filters.
        For different initial states across filters, set the dataframe column for corresponding to initial state
        with setInitialStateCol.

        :param value: pyspark.ml.linalg.Vector with size (stateSize)
        :return: KalmanFilter
        """
        return self._set(initialState=value)

    def setInitialStateCol(self, value):
        """
        Set the column corresponding to initial state vector. Overrides setInitialState setting.

        :param value: String
        :return: KalmanFilter
        """
        return self._set(initialStateCol=value)

    def setInitialCovariance(self, value):
        """
        Set the initial covariance matrix with dimensions (stateSize, stateSize)

        It will be applied to all states. If the state timeouts and starts receiving
        measurements after timeout, it will again start from this initial covariance vector. Default is identity matrix.
        :param value: pyspark.ml.linalg.Matrix with dimensions (stateSize, stateSize)
        :return: KalmanFilter
        """
        return self._set(initialCovariance=value)

    def setInitialCovarianceCol(self, value):
        """
        Set the column corresponding to initial covariance matrix. Overrides setInitialCovariance setting.

        :param value: String
        :return: KalmanFilter
        """
        return self._set(initialCovarianceCol=value)

    def setFadingFactor(self, value):
        """
        Fading factor for giving more weights to more recent measurements. If needed, it should be greater than one.
        Typically set around 1.01 ~ 1.05. Default is 1.0, which will result in equally weighted measurements.

        :param value: Float >= 1.0
        :return: KalmanFilter
        """
        return self._set(fadingFactor=value)

    def setProcessModel(self, value):
        """
        Set default value for process model matrix with dimensions (stateSize, stateSize) which governs
        state transition.

        Note that if this parameter is set through here, it will result in same process model for all filters &
        measurements. For different process models across filters or measurements, set a dataframe column for process
        model from setProcessModelCol.

        Default is identity matrix.

        :param value: pyspark.ml.linalg.Matrix with dimensions (stateSize, stateSize)
        :return: KalmanFilter
        """
        return self._set(processModel=value)

    def setProcessNoise(self, value):
        """
        Set default value for process noise matrix with dimensions (stateSize, stateSize).

        Note that if this parameter is set through here, it will result in same process noise for all filters &
        measurements. For different process noise values across filters or measurements, set a dataframe column
        for process noise from setProcessNoiseCol.

        Default is identity matrix.

        :param value: pyspark.ml.linalg.Matrix with dimensions (stateSize, StateSize)
        :return: KalmanFilter
        """
        return self._set(processNoise=value)

    def setMeasurementModel(self, value):
        """
        Set default value for measurement model matrix with dimensions (stateSize, measurementSize)
        which maps states to measurement.

        Note that if this parameter is set through here, it will result in same measurement model for all filters &
        measurements. For different measurement models across filters or measurements, set a dataframe column for
        measurement model from setMeasurementModelCol.

        Default value maps the first state value to measurements.

        :param value: pyspark.ml.linalg.Matrix with dimensions (stateSize, measurementSize)
        :return: KalmanFilter
        """
        return self._set(measurementModel=value)

    def setMeasurementNoise(self, value):
        """
        Set default value for measurement noise matrix with dimensions (measurementSize, measurementSize).

        Note that if this parameter is set through here, it will result in same measurement noise for all filters &
        measurements. For different measurement noise values across filters or measurements,
        set a dataframe column for measurement noise from setMeasurementNoiseCol.

        Default is identity matrix.

        :param value: pyspark.ml.linalg.Matrix with dimensions (measurementSize, measurementSize)
        :return: KalmanFilter
        """
        return self._set(measurementNoise=value)

    def setMeasurementCol(self, value):
        """
        Set the column corresponding to measurements.

        The vectors in the column should be of size (measurementSize). null values are allowed,
        which will result in only state prediction step.

        :param value: pyspark.ml.linalg.Vector with size measurementSize
        :return: KalmanFilter
        """
        return self._set(measurementCol=value)

    def setMeasurementModelCol(self, value):
        """
        Set the column for input measurement model matrices

        Measurement model matrices should have dimensions (stateSize, measurementSize)

        :param value: String
        :return: KalmanFilter
        """
        return self._set(measurementModelCol=value)

    def setProcessModelCol(self, value):
        """
        Set the column for input process model matrices.

        Process model matrices should have dimensions (stateSize, stateSize)

        :param value: String
        :return: KalmanFilter
        """
        return self._set(processModelCol=value)

    def setProcessNoiseCol(self, value):
        """
        Set the column for input process noise matrices.

        Process noise matrices should have dimensions (stateSize, stateSize)

        :param value: String
        :return: KalmanFilter
        """
        return self._set(processNoiseCol=value)

    def setMeasurementNoiseCol(self, value):
        """
        Set the column for input measurement noise matrices.

        Measurement noise matrices should have dimensions (measurementSize, measurementSize)

        :param value: String
        :return: KalmanFilter
        """
        return self._set(measurementNoiseCol=value)

    def setControlCol(self, value):
        """
        Set the column for input control vectors.

        Control vectors should have compatible size with control function (controlVectorSize). The product of
        control matrix & vector should produce a vector with stateSize. null values are allowed,
        which will result in state transition without control input.

        :param value: String
        :return: KalmanFilter
        """
        return self._set(controlCol=value)

    def setControlFunctionCol(self, value):
        """
        Set the column for input control matrices.

        Control matrices should have dimensions (stateSize, controlVectorSize). null values are allowed, which will
        result in state transition without control input

        :param value: String
        :return: KalmanFilter
        """
        return self._set(controlFunctionCol=value)

    def setCalculateLogLikelihood(self):
        """
        Optionally calculate loglikelihood of each measurement & add it to output dataframe. Loglikelihood is calculated
        from residual vector & residual covariance matrix.

        Not enabled by default.

        :return: KalmanFilter
        """
        return self._set(calculateLoglikelihood=True)

    def setCalculateMahalanobis(self):
        """
        Optionally calculate mahalanobis distance metric of each measurement & add it to output dataframe.
        Mahalanobis distance is calculated from residual vector & residual covariance matrix.

        Not enabled by default.

        :return: KalmanFilter
        """
        return self._set(calculateMahalanobis=True)

    def setOutputSystemMatrices(self):
        """
        Optionally add system matrices to output dataframe returned by the transformer.

        Default is false

        :return: KalmanFilter
        """
        return self._set(outputSystemMatrices=True)

    def setCalculateSlidingLikelihood(self):
        """
        Optionally calculate a sliding likelihood across consecutive measurements

        Default is false

        :return: KalmanFilter
        """
        return self._set(calculateSlidingLikelihood=True)

    def setSlidingLikelihoodWindow(self, value):
        """
        Set the param for number of consecutive measurements to include in the total likelihood calculation

        Default is 1

        :param value: Integer
        :return: KalmanFilter
        """
        return self._set(slidingLikelihoodWindow=value) \
            .setCalculateSlidingLikelihood()