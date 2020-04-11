#!/usr/bin/env python

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup


VERSION = "0.2.0"

with open("README.md") as f:
    long_description = f.read()

setup(
    name='artan',
    version=VERSION,
    description='Model-parallel bayesian filtering with Apache Spark.',
    long_description_content_type="text/markdown",
    long_description=long_description,
    author='Ozan Cicekci',
    author_email='ozancancicekci@gmail.com',
    url='https://github.com/ozancicek/artan',
    package_data={'': ['LICENSE']},
    include_package_data=True,
    packages=['artan',
              'artan.filter',
              'artan.smoother',
              'artan.state'],
    package_dir={
        'artan': 'python/artan',
        'artan.filter': 'python/artan/filter',
        'artan.smoother': 'python/artan/smoother',
        'artan.state': 'python/artan/state'
    },
    license='http://www.apache.org/licenses/LICENSE-2.0',
    keywords=['pyspark', 'kalman', 'filter', 'sparkml', 'structured', 'streaming'],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6'],
    zip_safe=False)
