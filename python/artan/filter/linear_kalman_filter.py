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
from pyspark.ml.util import JavaMLWritable

from artan.state import StatefulTransformer
from artan.utils import ArtanJavaMLReadable
from artan.filter.filter_params import (
    KalmanFilterParams, HasMultipleModelAdaptiveEstimationEnabled,
    HasMultipleModelMeasurementWindowDuration)


class LinearKalmanFilter(StatefulTransformer, KalmanFilterParams, HasMultipleModelAdaptiveEstimationEnabled,
                         HasMultipleModelMeasurementWindowDuration, ArtanJavaMLReadable, JavaMLWritable):
    """
    Linear Kalman Filter, implemented with a stateful spark Transformer for running parallel filters /w spark
    dataframes. Transforms an input dataframe of noisy measurements to dataframe of state estimates using stateful
    spark transformations, which can be used in both streaming and batch applications.

    Assuming a state vector :math:`x_k` with size `stateSize`, and measurements vector :math:`z_k`
    with size `measurementSize`, below parameters can be specified.

    - :math:`F_k`, process model, matrix with dimensions `stateSize` x `stateSize`
    - :math:`H_k`, measurement model, matrix with dimensions `stateSize` x `measurementSize`
    - :math:`Q_k`, process noise covariance, matrix with dimensions `stateSize` x `stateSize`
    - :math:`R_k`, measurement noise covariance, matrix with dimensions `measurementSize` x `measurementSize`
    - :math:`u_k`, optional control vector, vector with size `controlSize`
    - :math:`B_k`, optional control model, matrix with dimensions `stateSize` x `controlSize`

    Linear Kalman Filter will predict & estimate the state according to following state and measurement equations.

    .. math::

        x_k &= F_k x_{k-1} + B_k u_k + v_k \\

        z_k &= H_k x_k + w_k

    Where :math:`v_k` and :math:`w_k` are noise vectors drawn from zero mean,
    :math:`Q_k` and :math:`R_k` covariance distributions.

    The default values of system matrices will not give you a functioning filter, but they will be initialized
    with reasonable values given the state and measurement sizes. All of the inputs to the filter can
    be specified with a dataframe column which will allow you to have different value across measurements/filters,
    or you can specify a constant value across all measurements/filters.
    """
    def __init__(self):
        super(LinearKalmanFilter, self).__init__()
        self._java_obj = self._new_java_obj("com.github.ozancicek.artan.ml.filter.LinearKalmanFilter", self.uid)


    def setMultipleModelMeasurementWindowDuration(self, value):
        """
        Optionally set the window duration for grouping measurements in same window for MMAE filter aggregation.
        Could be used for limiting the state on streaming if event time column is set.

        :param value: String
        :return: LinearKalmanFilter
        """
        return self._set(multipleModelMeasurementWindowDuration=value)

    def setEnableMultipleModelAdaptiveEstimation(self):
        """
        Enable MMAE output mode

        :return: LinearKalmanFilter
        """
        return self._set(multipleModelAdaptiveEstimationEnabled=True)

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        py_stage = LinearKalmanFilter()
        py_stage._java_obj = java_stage
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()
        return py_stage