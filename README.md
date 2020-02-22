# Artan
[![Build Status](https://travis-ci.com/ozancicek/artan.svg?branch=master)](https://travis-ci.com/ozancicek/artan)
[![codecov](https://codecov.io/gh/ozancicek/artan/branch/master/graph/badge.svg)](https://codecov.io/gh/ozancicek/artan)


Model-parallel bayesian filtering with Apache Spark. It supports mainly running parallel Kalman filters
(extended, unscented, etc,..), and various other filters. Both structured streaming & batch processing mode is supported
by leveraging arbitrary stateful transformation capabilities of Spark DataFrames.