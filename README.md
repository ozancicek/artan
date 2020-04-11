# Artan
[![Build Status](https://travis-ci.com/ozancicek/artan.svg?branch=master)](https://travis-ci.com/ozancicek/artan)
[![codecov](https://codecov.io/gh/ozancicek/artan/branch/master/graph/badge.svg)](https://codecov.io/gh/ozancicek/artan)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.ozancicek/artan_2.11)](https://mvnrepository.com/artifact/com.github.ozancicek/artan)
[![PyPI](https://img.shields.io/pypi/v/artan)](https://pypi.org/project/artan/)
[![Documentation Status](https://readthedocs.org/projects/artan/badge/?version=latest)](https://artan.readthedocs.io/en/latest/?badge=latest)


Model-parallel bayesian filtering with Apache Spark.

- [Overview](#overview)
- [Download](#download)
- [Docs and Examples](#docs-and-examples)
- [Usage](#usage)

## Overview

This library provides supports for running various bayesian filters in parallel with Apache Spark. Uses arbitrary
stateful transformation capabilities of Spark DataFrames to define model-parallel bayesian filters. Therefore, it
is suitable for latent state estimation of many similar small scale systems rather than a big single system.

Both structured streaming & batch processing modes are supported. Implemented filters extend SparkML Transformers, so
you can transform a DataFrame of measurements to a DataFrame of estimated states with Kalman filters
(extended, unscented, etc,..) and various other filters as a part of your SparkML Pipeline.

Artan requires Scala 2.11, Spark 2.4+ and Python 3,6+


## Download

This project has been published to the Maven Central Repository. When submitting jobs on your cluster, you can use
`spark-submit` with `--packages` parameter to download all required dependencies including python packages.

    spark-submit --packages='com.github.ozancicek:artan_2.11:0.1.0'

For SBT:

    libraryDependencies += "com.github.ozancicek" %% "artan" % "0.1.0"

For python:

    pip install artan

Note that pip will only install the python dependencies. To submit pyspark jobs, `--packages='com.github.ozancicek:artan_2.11:0.1.0'` argument should be specified in order to download necessary jars.


## Docs and Examples

Visit [docs](https://artan.readthedocs.io/) and [examples](https://github.com/ozancicek/artan/blob/master/examples/src/main) for all sample scripts.

### Streaming examples
- Local linear trend filtering with Linear Kalman Filter ([python](https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/lkf_rate_source_llt.py), [scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/LKFRateSourceLLT.scala))
- Recursive least squares ([python](https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/rls_rate_source_ols.py), [scala](examples/src/main/scala/com/ozancicek/artan/examples/streaming/RLSRateSourceOLS.scala))
- Nonlinear estimation with Extended Kalman Filter ([scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/EKFRateSourceGLMLog.scala))
- Nonlinear estimation with Unscented Kalman Filter ([scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/UKFRateSourceGLMLog.scala))
- Multiple-Model Adaptive estimation ([scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/MMAERateSourceOLS.scala))
