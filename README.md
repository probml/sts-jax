# sts-jax
Structural Time Series (STS) in JAX

This library builds on
[Dynamax](https://github.com/probml/dynamax/tree/main/dynamax) library
for state-space models in JAX.
It has a similar to design to [tfp.sts](https://www.tensorflow.org/probability/api_docs/python/tfp/sts),
but is built entirely in JAX.
The model parameters can be fit using maximum likelihood estimation (MLE),
or using Bayesian inference;
the latter can use variational inference or HMC
(implemented by [blackjax](https://github.com/blackjax-devs/blackjax)).


There is also a preliminary implementation of the
[causal impact](https://google.github.io/CausalImpact/) method.
This has a similar to design to [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact),
but is built entirely in JAX.

## Documentation

To install the latest development branch:

``` {.console}
pip install git+https://github.com/probml/dynamax.git
```

## What are structural time series (STS) models?

The STS model is a linear state space model with a specific structure. In particular,
the latent state $z_t$ is a composition of states of all latent components:

$$z_t = [c_{1, t}, c_{2, t}, ...]$$

where $c_{i,t}$ is the state of latent component $c_i$ at time step $t$.

The STS model takes the form:

$$y_t = H_t z_t + u_t + \epsilon_t, \qquad  \epsilon_t \sim \mathcal{N}(0, \Sigma_t)$$
$$z_{t+1} = F_t z_t + R_t \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_t)$$

where

* $H_t$: emission matrix, which sums up the contributions of all latent components.
* $u_t$: is the contribution of the regression component.
* $F_t$: transition matrix of the latent dynamics of the state space model
* $R_t$: the selection matrix, which is a subset of columns of base vector $e_i$, converting
    the non-singular covariance matrix into the (possibly singular) covariance matrix of
    the latent state $z_t$.
* $Q_t$: non-singular covariance matrix of the latent state, so the dimension of $Q_t$
        can be smaller than the dimension of $z_t$.

The covariance matrix of the latent dynamics model takes the form $R Q R^T$, where $Q$ is
a non-singular matrix (block diagonal), and $R$ is the selecting matrix.

More information of STS models can be found in these books:

> -   \"Machine Learning: Advanced Topics\", K. Murphy, MIT Press 2023.
>     Available at <https://probml.github.io/pml-book/book2.html>.
> -   \"Time Series Analysis by State Space Methods (2nd edn)\", James Durbin, Siem Jan Koopman,
>     Oxford University Press, 2012.

## Example usage

In this library, an STS model is constructed by providing the observed time series and specifying a list of
components and the distribution family of the observation. This library implements
common STS components including **local linear trend** component, **seasonal** component, 
**cycle** component, **autoregressive** component, and **regression** component. More components
will be added in the future. The observed time series can follow either the **Gaussian**
distribution or the **Poisson** distribution. 

Internally, the STS model is converted to the corresponding state space model (SSM) and inference
and learning of parameters are performed on the SSM.

If the observation $Y_t$ follows a Gaussian distribution, the inference of latent variables
$Z_{1:T}$ is based on the Kalman filter and the Kalman smoother algorithm implemented in the 
library dynamax.

Alternatively, if the observation $Y_t$ follows Poisson distribution, the inference of the
latent variables $Z_{1:t}$ is based on the algorithm 'conditional moment Gaussian filtering
(smoothing)' implemented in the library dynamax.
 
The marginal likelihood of $y_{1:T}$ conditioned on parameters can be evaluated as a 
byproduct during the inference. Using autograd of JAX, the parameters of the STS model
can be learned via **MLE**(based on SGD implemented in the library 'optax'), **VI**,
or **HMC**(from the library 'blackjax').

Below we illustrate the high level (object-oriented) API for the case of an STS with Gaussian
observations. (See [this folder](./sts_jax/structural_time_series/demos/) for runnable versions
of the following demos of STS models, and see [this file](./sts_jax/causal_impact/causal_impact_demo.ipynb)
for a runnable version of the demo of the causal impact model).

## Electricity usage example

This example is adapted from the [blog](https://blog.tensorflow.org/2019/03/structural-time-series-modeling-in.html?utm_source=www.tensorflow.org&utm_medium=referral&_gl=1*ejl315*_ga*OTYwMTY3MzQxLjE2Njg1ODM4OTY.*_ga_W0YLR4190T*MTY3MDkwNDczMy4yMi4xLjE2NzA5MDQ3MzQuMC4wLjA.&max-results=20)
in the sts library of tensorflow_probability (TFP).

The problem of interest is to forecast electricity demand in Victoria, Australia.
The dataset contains hourly record of electricity demand and temperature measurements 
from the first 8 weeks of 2014. The records of temperature will be used to model the trend 
of the electricity demands via a **regression** component. The following plot is the training
set of the data, which contains measurements in the first 6 weeks.

<p align="center">
<img src="./sts_jax/figures/electr_obs.png" alt="drawing" style="width:600px;"/>
<p>

```python
import sts_jax.structural_time_series.sts_model as sts

# The STS model has two seasonal components, one autoregressive component,
# and one regression component.
hour_of_day_effect = sts.SeasonalDummy(num_seasons=24,
                                       name='hour_of_day_effect')
day_of_week_effect = sts.SeasonalTrig(num_seasons=7, num_steps_per_season=24,
                                      name='day_of_week_effect')
temperature_effect = sts.LinearRegression(dim_covariates=1, add_bias=True,
                                          name='temperature_effect')
autoregress_effect = sts.Autoregressive(order=1,
                                        name='autoregress_effect')

# The STS model is constructed by providing the observed time series,
# specifying a list of components and the distribution family of the observations.
model = sts.StructuralTimeSeries(
    [hour_of_day_effect, day_of_week_effect, temperature_effect, autoregress_effect],
    obs_time_series,
    obs_distribution='Gaussian',
    covariates=temperature_training_data)

# Perform the MLE estimation of parameters via SGD implemented in dynamax library.
opt_param, _losses = model.fit_mle(obs_time_series,
                                   covariates=temperature_training_data,
                                   num_steps=2000)

# The 'forecast' method samples the future means and future observations from the
# predictive distribution, conditioned on the parameters of the model. 
forecast_means, forecasts = model.forecast(opt_param,
                                           obs_time_series,
                                           num_forecast_steps,
                                           past_covariates=temperature_training_data,
                                           forecast_covariates=temperature_predict_data)
```

The following plot is the mean values and interval estimations of the forecast.
<p align="center">
<img src="./sts_jax/figures/electr_forecast.png" alt="drawing" style="width:600px;"/>
<p>


## Time series with Poisson distribution.
This library can also fit time series with discrete observations following the Poisson 
distribution. Internally, the inference of latent states $Z_{1:T}$ the corresponding SSM
is based on the algorithm 'conditional moment Gaussian filtering (smoothing)' implemented
in the library dynamax. An STS model for a Poisson-distributed time series can be constructed
simply by specifying observation distribution to be 'Poisson'. Everything else is the same
with the Gaussian case.

```python
import sts_jax.structural_time_series.sts_model as sts

# This example uses a synthetic dataset and the STS model contains only a
# local linear trend component.
trend = sts.LocalLinearTrend()
model = sts.StructuralTimeSeries([trend],
                                 obs_distribution='Poisson',
                                 obs_time_series=counts_training)

# Fit the model using HMC algorithm
param_samples, _log_probs = model.fit_hmc(num_samples=200,
                                          obs_time_series=counts_training)

# Forecast into the future given samples of parameters returned by the HMC algorithm.
forecasts = model.forecast(param_samples, obs_time_series, num_forecast_steps)[1]
```
<p align="center">
<img src="./sts_jax/figures/poisson_forecast.png" alt="drawing" style="width:600px;"/>
<p>

### Compare the sampling speed with the implementation in TFP.

The STS library of TFP also support fitting STS models for Poisson-distributed
time series (see [this blog](https://www.tensorflow.org/probability/examples/STS_approximate_inference_for_models_with_non_Gaussian_observations)).
The learning and inference is performed by running HMC on the joint
model of latent states $Z_{1:T}$ and the parameters, conditioned 
on observation $Y_{1:T}$. Since the dimension of the state space of HMC grows linearly
with the length of the time series to be fitted, the implementation will be inefficient
when $T$ is relatively large. This is not a big issue for the HMC algorithm in this library,
since the inference is of $Z_{1:T}$ here is performed via approximate filtering, the state 
space of the HMC algorithm is just the model parameters, no matter how large is $T$.

A comparison of the fitting time of HMC implemented in this library and in STS library
of TFP is demonstrated in the following plot. The burnin steps of HMC in the TFP-STS
implementation is adjusted such that the forecast error of the two implementations
are comparable. It can be seen that the running time of HMC in TFP-STS increases linearly
with $T$, and the running time of the HMC algorithm in this library is not affected as
much.

<p align="center">
<img src="./sts_jax/figures/comparison.png" alt="drawing" style="width:600px;"/>
<p>

## Inference of Causal Impact

The inference of causal impact is implemented on top of the STS module.
The following demo is an example of inferring causal impact where an intervention on
the time series $Y_{1:T}$ happened at time step $t=70$, and another time series
$X_{1:T}$, which is not affected by the intervention, is used as a covariate in
modeling the trend of $Y_{1:T}$.


<p align="center">
<img src="./sts_jax/figures/causal_obs.png" alt="drawing" style="width:600px;"/>
<p>

```python
from sts_jax.causal_impact.causal_impact import causal_impact

# The causal impact is inferred by providing the target time series and covariates,
# specifying the intervention time and the distribution family of the observation.
# If the STS model is not given, an STS model with only a local linear trend component
# in addition to the regression component is constructed by default internally.
impact = causal_impact(obs_time_series,
                       intervention_timepoint,
                       'Gaussian',
                       covariates,
                       sts_model=None)

# The visualization of the causal impact is provided by the 'plot' method
# of the object returned by the function 'causal_impact'.
impact.plot()
```
<p align="center">
<img src="./sts_jax/figures/causal_forecast.png" alt="drawing" style="width:600px;"/>
<p>

```python
# The object returned by the function 'causal_impact' also provides 
# a method that prints the summary table of inferred causal effect.
impact.print_summary()

# The output has the following form:
```
```python
Posterior inference of the causal impact:

                               Average            Cumulative     
Actual                          129.93             3897.88       

Prediction (s.d.)           120.01 (2.04)      3600.42 (61.31)   
95% CI                     [114.82, 123.07]   [3444.72, 3692.09] 

Absolute effect (s.d.)       9.92 (2.04)        297.45 (61.31)   
95% CI                      [6.86, 15.11]      [205.78, 453.16]  

Relative effect (s.d.)      8.29% (1.89%)       8.29% (1.89%)    
95% CI                     [5.57%, 13.16%]     [5.57%, 13.16%]   

Posterior tail-area probability p: 0.0050
Posterior prob of a causal effect: 99.50%
```

## Contributing

Please see [this page](https://github.com/probml/dynamax/blob/main/CONTRIBUTING.md) for details
on how to contribute.

## About

MIT License. 2022
