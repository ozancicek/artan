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

from pyspark.ml.common import inherit_doc
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import JavaMLWritable
from artan.utils import ArtanJavaMLReadable
from artan.state import StatefulTransformer
from artan.filter.filter_params import KalmanFilterParams


class HasFixedLag(Params):
    """
    Mixin for param for fixed lag
    """

    fixedLag = Param(
        Params._dummy(),
        "fixedLag", "Fixed lag", typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasFixedLag, self).__init__()

    def getFixedLag(self):
        """
        Gets the value of fixed lag or its default value.
        """
        return self.getOrDefault(self.fixedLag)


@inherit_doc
class LinearKalmanSmoother(StatefulTransformer, KalmanFilterParams, HasFixedLag,
                           ArtanJavaMLReadable, JavaMLWritable):
    """
    Fixed lag linear kalman smoother using Rauch-Tung-Striebel method. The smoother is implemented with a
    stateful spark transformer for running parallel smoother /w spark dataframes.
    Transforms an input dataframe of noisy measurements to dataframe of state estimates using
    stateful spark transformations, which can be used in both streaming and batch applications.

    At a time step k and a fixed lag N, the fixed lag linear kalman smoother computes the state estimates of a linear
    kalman filter based on all measurements made between step k and step k-t. For each time step k >= N, the smoother
    outputs an estimate for all the time steps between k and k-N. When k < N, the smoother doesn't output any estimates.
    As a result, the memory requirements of this filter is N times of a linear kalman filter. Since the smoother
    outputs multiple estimates for a single measurement, it is advised to set event time column
    of the measurements with setEventTimeCol.
    """

    java_class = "com.github.ozancicek.artan.ml.smoother.LinearKalmanSmoother"

    def __init__(self):
        super(LinearKalmanSmoother, self).__init__()
        self._java_obj = self._new_java_obj(LinearKalmanSmoother.java_class,
                                            self.uid)

    def setFixedLag(self, value):
        """
        Sets the smoother fixed lag

        Default is 2.
        :param value: Int
        :return: LinearKalmanSmoother
        """
        return self._set(fixedLag=value)

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        py_stage = LinearKalmanSmoother()
        py_stage._java_obj = java_stage
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()
        return py_stage