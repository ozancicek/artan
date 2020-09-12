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

from pyspark.ml.util import MLReadable, JavaMLReader


class ArtanJavaMLReadable(MLReadable):
    """
    ML readable class for packages under com.github.ozancicek.artan
    """

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return _ArtanJavaMLReader(cls)


class _ArtanJavaMLReader(JavaMLReader):
    """
    Custom JavaMLReader which overrides default org.apache.spark class location

    """
    @classmethod
    def _java_loader_class(cls, clazz):
        """
        Returns the full class name of the Java ML instance. The default
        implementation replaces "pyspark" by "org.apache.spark", so override it to return
        this projects class path
        """
        # Remove the last module
        python_loc = clazz.__module__.split(".")[1]
        java_package = "com.github.ozancicek.artan.ml." + python_loc + "." + clazz.__name__
        return java_package