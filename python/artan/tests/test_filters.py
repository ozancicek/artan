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

from artan.testing.sql_utils import ReusedSparkTestCase
from artan.filter import RecursiveLeastSquaresFilter
from pyspark.ml.linalg import Vectors
import numpy as np


class RLSTests(ReusedSparkTestCase):

    def test_rls(self):
        df = self.spark.createDataFrame(
            [(1.0, Vectors.dense(0.0, 5.0)),
             (0.0, Vectors.dense(1.0, 2.0)),
             (1.0, Vectors.dense(2.0, 1.0)),
             (0.0, Vectors.dense(3.0, 3.0)), ], ["label", "features"])
        rls = RecursiveLeastSquaresFilter(2)
        model = rls.transform(df).filter("stateIndex=4").collect()
        state = model[0].state.values
        expected = np.array([5.31071176e-09, 1.53846148e-01])

        np.testing.assert_array_almost_equal(state, expected)
