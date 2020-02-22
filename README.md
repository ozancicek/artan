# Artan
[![Build Status](https://travis-ci.com/ozancicek/artan.svg?branch=master)](https://travis-ci.com/ozancicek/artan)
[![codecov](https://codecov.io/gh/ozancicek/artan/branch/master/graph/badge.svg)](https://codecov.io/gh/ozancicek/artan)


Model-parallel bayesian filtering with Apache Spark. Both structured streaming & batch processing modes are
supported by leveraging arbitrary stateful transformation capabilities of Spark DataFrames. Allows you to transform
a DataFrame of measurements to a DataFrame of estimated states by running parallel Kalman filters
(extended, unscented, etc,..) and various other filters.