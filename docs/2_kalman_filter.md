# Documentation


This tutorial will show you a streaming Kalman filter example with this library. The example consists of
online training of a model-parallel Recursive Least Squares, using a Kalman filter with
spark structured streaming.


## Online Recursive Least Squares with Kalman filter

Recursive estimation of least squares can be easily done with a Kalman filter. Using state-space
representation, the following linear model:


![linmod](https://latex.codecogs.com/svg.latex?%5C%5C%20Y_t%20%3D%20%5Cbeta%20X_t%20&plus;%20%5Cepsilon%20%3A%20%5Cepsilon%20%24%5Csim%24%20N%280%2C%20R%29%20%5Cquad%20t%3D%201%2C%202%2C%20...%20T%20%5C%5C)

Can be represented in state-space form by:

![statespace](https://latex.codecogs.com/svg.latex?%5C%5C%20V_t%20%3D%20A_t%20V_%7Bt%20-%201%7D%20&plus;%20q_%7Bt%7D%3A%20q_t%20%24%5Csim%24%20N%280%2C%20Q%29%20%5Cquad%20%28state%20%5C%20process%20%5C%20equation%29%20%5C%5C%20Z_t%20%3D%20H_t%20V_t%20&plus;%20r_t%3A%20r_t%20%24%5Csim%24%20N%280%2C%20R%29%20%5Cquad%20%28measurement%20%5C%20equation%29%20%5C%5C%20%5C%5C%20A_t%20%3D%20I%5C%5C%20H_t%20%3D%20X_t%5C%5C%20q_t%20%3D%200)

At each time step `t`, the state would give an estimate of the model parameters.

#### Scala

See [examples](/examples/src/main/scala/com/ozancicek/artan/examples/streaming/LKFRateSourceOLS.scala) for the full code

#### Python

See [examples](/examples/src/main/python/streaming/lkf_rate_source_ols.py) for the full code
