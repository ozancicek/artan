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