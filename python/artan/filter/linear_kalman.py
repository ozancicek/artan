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
from artan.filter.filter_params import *


class LinearKalmanFilter(StatefulTransformer, HasInitialState, HasInitialCovariance, HasProcessModel,
                         HasFadingFactor, HasMeasurementModel, HasMeasurementNoise, HasProcessNoise,
                         HasMeasurementCol, HasMeasurementModelCol, HasMeasurementNoiseCol,
                         HasProcessModelCol, HasProcessNoiseCol, HasControlCol, HasControlFunctionCol,
                         HasCalculateMahalanobis, HasCalculateLoglikelihood):
    """
    Linear Kalman Filter, implemented with a stateful spark Transformer for running parallel filters /w spark
    dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
    spark transormations, which can be used in both streaming and batch applications.

    Assuming a state (x_k) with size n_s, and measurements (z_k) with size n_m,
    following parameters should be specified;

    - F_k, process model, matrix with dimensions (n_s, n_s)
    - H_k, measurement model, matrix with dimensions (n_s, n_m)
    - Q_k, process noise covariance, matrix with dimensions (n_s, n_s)
    - R_k, measurement noise covariance, matrix with dimensions (n_m, n_m)
    - B_k, optional control model, matrix with dimensions (n_s, n_control)
    - u_k, optional control vector, vector with size (n_control)

    Linear Kalman Filter will predict & estimate the state according to following equations

    State prediction:
    x_k = F_k * x_k-1 + B_k * u_k + w_k

    Measurement incorporation:
    z_k = H_k * x_k + v_k

    Where v_k and w_k are noise vectors drawn from zero mean, Q_k and R_k covariance distributions.

    The default values of system matrices will not give you a functioning filter, but they will be initialized
    with reasonable values given the state and measurement sizes. All of the inputs to the filter can
    be specified with a dataframe column which will allow you to have different value across measurements/filters,
    or you can specify a constant value across all measurements/filters.
    """
    def __init__(self, stateSize, measurementSize):
        super(LinearKalmanFilter, self).__init__()
        self._java_obj = self._new_java_obj("com.ozancicek.artan.ml.filter.LinearKalmanFilter",
                                            stateSize, measurementSize, self.uid)

    def setInitialState(self, value):
        """
        Set the initial state vector with size (stateSize).

        It will be applied to all states. If the state timeouts and starts receiving
        measurements after timeout, it will again start from this initial state vector. Default is zero.

        :param value: pyspark.ml.linalg.Vector with size (stateSize)
        :return: LinearKalmanFilter
        """
        return self._set(initialState=value)

    def setInitialCovariance(self, value):
        """
        Set the initial covariance matrix with dimensions (stateSize, stateSize)

        It will be applied to all states. If the state timeouts and starts receiving
        measurements after timeout, it will again start from this initial covariance vector. Default is identity matrix.
        :param value: pyspark.ml.linalg.Matrix with dimensions (stateSize, stateSize)
        :return: LinearKalmanFilter
        """
        return self._set(initialCovariance=value)

    def setFadingFactor(self, value):
        """
        Fading factor for giving more weights to more recent measurements. If needed, it should be greater than one.
        Typically set around 1.01 ~ 1.05. Default is 1.0, which will result in equally weighted measurements.

        :param value: Float >= 1.0
        :return: LinearKalmanFilter
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
        :return:
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
        :return: LinearKalmanFilter
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
        :return: LinearKalmanFilter
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
        :return: LinearKalmanFilter
        """
        return self._set(measurementNoise=value)

    def setMeasurementCol(self, value):
        """
        Set the column corresponding to measurements.

        The vectors in the column should be of size (measurementSize). null values are allowed,
        which will result in only state prediction step.

        :param value: pyspark.ml.linalg.Vector with size measurementSize
        :return: LinearKalmanFilter
        """
        return self._set(measurementCol=value)

    def setMeasurementModelCol(self, value):
        """
        Set the column for input measurement model matrices

        Measurement model matrices should have dimensions (stateSize, measurementSize)

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(measurementModelCol=value)

    def setProcessModelCol(self, value):
        """
        Set the column for input process model matrices.

        Process model matrices should have dimensions (stateSize, stateSize)

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(processModelCol=value)

    def setProcessNoiseCol(self, value):
        """
        Set the column for input process noise matrices.

        Process noise matrices should have dimensions (stateSize, stateSize)

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(processNoiseCol=value)

    def setMeasurementNoiseCol(self, value):
        """
        Set the column for input measurement noise matrices.

        Measurement noise matrices should have dimensions (measurementSize, measurementSize)

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(measurementNoiseCol=value)

    def setControlCol(self, value):
        """
        Set the column for input control vectors.

        Control vectors should have compatible size with control function (controlVectorSize). The product of
        control matrix & vector should produce a vector with stateSize. null values are allowed,
        which will result in state transition without control input.

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(controlCol=value)

    def setControlFunctionCol(self, value):
        """
        Set the column for input control matrices.

        Control matrices should have dimensions (stateSize, controlVectorSize). null values are allowed, which will
        result in state transition without control input

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(controlFunctionCol=value)

    def setCalculateLogLikelihood(self):
        """
        Optionally calculate loglikelihood of each measurement & add it to output dataframe. Loglikelihood is calculated
        from residual vector & residual covariance matrix.

        Not enabled by default.

        :return: LinearKalmanFilter
        """
        return self._set(calculateLoglikelihood=True)

    def setCalculateMahalanobis(self):
        """
        Optinally calculate mahalanobis distance metric of each measuremenet & add it to output dataframe.
        Mahalanobis distance is calculated from residual vector & residual covariance matrix.

        Not enabled by default.

        :return: LinearKalmanFilter
        """
        return self._set(calculateMahalanobis=True)