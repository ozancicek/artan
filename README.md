# Artan
[![Build Status](https://travis-ci.com/ozancicek/artan.svg?branch=master)](https://travis-ci.com/ozancicek/artan)
[![codecov](https://codecov.io/gh/ozancicek/artan/branch/master/graph/badge.svg)](https://codecov.io/gh/ozancicek/artan)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.ozancicek/artan_2.11)](https://mvnrepository.com/artifact/com.github.ozancicek/artan)
[![PyPI](https://img.shields.io/pypi/v/artan)](https://pypi.org/project/artan/)
[![Documentation Status](https://readthedocs.org/projects/artan/badge/?version=latest)](https://artan.readthedocs.io/en/latest/?badge=latest)


Model-parallel online latent state estimation with Apache Spark.

- [Overview](#overview)
- [Download](#download)
- [Docs and Examples](#docs-and-examples)
- [Usage](#usage)

## Overview

This library provides supports for model-parallel latent state estimation with Apache Spark, with a focus on online
learning compatible with structured streaming. Mainly developed for time series estimation of latent variables of many small
scale systems, this library could fit to your use case if you're looking for:

- **Model-parallelism.** Model-parallelism is the main mode of parallelism, such as training multiple similar time series 
models from online measurements/multiple sensors, or same models with different priors/hyperparameters etc,..
- **Online learning.** Model parameters are updated sequentially with measurements with a single pass. The state used
by the algorithms are bounded with #models and model parameters.
- **Latent state estimation.** Focusing on methods for hidden state estimation, implemented methods include solutions for
filtering (Kalman filters, EKF, UKF, Multiple-Model Adaptive filters, etc..) problems, smoothing (RTS) problems,
finite mixture models (Multivariate Gaussian, Poisson, Bernoulli, etc,..). 

Artan requires Scala 2.11, Spark 2.4+ and Python 3,6+


## Download

This project has been published to the Maven Central Repository. When submitting jobs on your cluster, you can use
`spark-submit` with `--packages` parameter to download all required dependencies including python packages.

    spark-submit --packages='com.github.ozancicek:artan_2.11:0.3.0'

For SBT:

    libraryDependencies += "com.github.ozancicek" %% "artan" % "0.3.0"

For python:

    pip install artan

Note that pip will only install the python dependencies. To submit pyspark jobs, `--packages='com.github.ozancicek:artan_2.11:0.3.0'` argument should be specified in order to download necessary jars.


## Docs and Examples

Visit [docs](https://artan.readthedocs.io/) to get started and see [examples](https://github.com/ozancicek/artan/blob/master/examples/src/main) for all sample scripts.

### Structured streaming examples
- Local linear trend filtering with Linear Kalman Filter ([python](https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/lkf_rate_source_llt.py), [scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/LKFRateSourceLLT.scala))
- Recursive least squares ([python](https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/rls_rate_source_ols.py), [scala](examples/src/main/scala/com/ozancicek/artan/examples/streaming/RLSRateSourceOLS.scala))
- Nonlinear estimation with Extended Kalman Filter ([scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/EKFRateSourceGLMLog.scala))
- Nonlinear estimation with Unscented Kalman Filter ([scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/UKFRateSourceGLMLog.scala))
- Multiple-Model Adaptive estimation ([scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/MMAERateSourceOLS.scala))
- Online Gaussian Mixture Model ([python](https://github.com/ozancicek/artan/blob/master/examples/src/main/python/streaming/gmm_rate_source.py), [scala](https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/github/ozancicek/artan/examples/streaming/GMMRateSource.scala))
