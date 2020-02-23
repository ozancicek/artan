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
from pyspark.ml.wrapper import JavaTransformer


class HasStateKeyCol(Params):
    """
    Mixin for param for state key column.
    """

    stateKeyCol = Param(
        Params._dummy(),
        "stateKeyCol",
        "State key column. State keys uniquely identify the each state in stateful transformers," +
        "thus controlling the number of states and the degree of parallelization",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasStateKeyCol, self).__init__()

    def getStateKeyCol(self):
        """
        Gets the value of state key column or its default value.
        """
        return self.getOrDefault(self.stateKeyCol)


class HasEventTimeCol(Params):
    """
    Mixin for param for event time column.
    """

    eventTimeCol = Param(
        Params._dummy(),
        "eventTimeCol",
        "Column marking the event time of the received measurements",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasEventTimeCol, self).__init__()

    def getEventTimeCol(self):
        """
        Gets the value of event time column or its default value.
        """
        return self.getOrDefault(self.eventTimeCol)


class HasWatermarkDuration(Params):
    """
    Mixin for param for watermark duration.
    """

    watermarkDuration = Param(
        Params._dummy(),
        "watermarkDuration",
        "Watermark duration",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasWatermarkDuration, self).__init__()

    def getWatermarkDuration(self):
        """
        Gets the value of watermark duration or its default value.
        """
        return self.getOrDefault(self.watermarkDuration)


class HasStateTimeoutMode(Params):
    """
    Mixin for param for state timeout mode for clearing states without updates, one of "none", "process" or "event".
    """

    timeoutMode = Param(
        Params._dummy(),
        "timeoutMode",
        "Timeout mode for clearing the states that didn't receive measurements.",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasStateTimeoutMode, self).__init__()

    def getTimeoutMode(self):
        """
        Gets the value of timeout mode or its default value.
        """
        return self.getOrDefault(self.timeoutMode)


class HasStateTimeoutDuration(Params):
    """
    Mixin for param for state timeout duration.
    """

    stateTimeoutDuration = Param(
        Params._dummy(),
        "stateTimeoutDuration",
        "Duration to wait before timing out the state",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasStateTimeoutDuration, self).__init__()

    def getStateTimeoutDuration(self):
        """
        Gets the value of state timeout duration or its default value.
        """
        return self.getOrDefault(self.stateTimeoutDuration)


class StatefulTransformer(JavaTransformer,
                          HasStateKeyCol, HasEventTimeCol, HasWatermarkDuration,
                          HasStateTimeoutDuration, HasStateTimeoutMode):
    """
    Base mixin for stateful transformations
    """
    def setStateKeyCol(self, value):
        """
        Sets the state key column. Each value in the column should uniquely identify a stateful transformer. Each
        unique value will result in a separate state.
        """
        return self._set(stateKeyCol=value)

    def setStateTimeoutMode(self, value):
        """
        Sets the state timeout mode. Supported values are 'none', 'process' and 'event'. Enabling state timeout will
        clear the state after a certain timeout duration which can be set. If a state receives measurements after
        it times out, the state will be initialized as if it received no measurements.

        - 'none': No state timeout, state is kept indefinitely.

        - 'process': Process time based state timeout, state will be cleared if no measurements are received for
            a duration based on processing time. Effects all states. Timeout duration must be set with
            setStateTimeoutDuration.

        - 'event': Event time based state timeout, state will be cleared if no measurements are recieved for a duration
            based on event time determined by watermark. Effects all states. Timeout duration must be set with
            setStateTimeoutDuration. Additionally, event time column and it's watermark duration must be set with
            setEventTimeCol and setWatermarkDuration. Note that this will result in dropping measurements occuring later
            than the watermark.

        Default is 'none'
        """
        return self._set(timeoutMode=value)

    def setEventTimeCol(self, value):
        """
        Sets the event time column in the input DataFrame for event time based state timeout.
        """
        return self._set(eventTimeCol=value)

    def setWatermarkDuration(self, value):
        """
        Set the watermark duration for all states, only valid when state timeout mode is 'event'.
        Must be a valid duration string, such as '10 minutes'.
        """
        return self._set(watermarkDuration=value)


    def setStateTimeoutDuration(self, value):
        """
        Sets the state timeout duration for all states, only valid when state timeout mode is not 'none'.
        Must be a valid duration string, such as '10 minutes'.
        """
        return self._set(stateTimeoutDuration=value)
