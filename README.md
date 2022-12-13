# sts-jax
Structural Time Series (STS) in JAX

This library builds on
[Dynamax](https://github.com/probml/dynamax/tree/main/dynamax) library
for state-space models in JAX.
It has a similar to design to [tfp.sts](https://www.tensorflow.org/probability/api_docs/python/tfp/sts), but is built entirely in JAX.
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

$$y_t = H_t z_t + u_t \epsilon_t, \qquad  \epsilon_t \sim \mathcal{N}(0, \Sigma_t)$$
$$z_{t+1} = F_t z_t + R_t \eta_t, \qquad eta_t \sim \mathcal{N}(0, Q_t)$$

where

* $H_t$: emission matrix, which sums up the contributions of all latent components.
* $u_t$: is the contribution of the regression component.
* $F_t$: transition matrix of the latent dynamics
* $R_t$: the selection matrix, which is a subset of clumnes of base vector $I$, converting
    the non-singular covariance matrix into a (possibly singular) covariance matrix for
    the latent state $z_t$.
* $Q_t$: non-singular covariance matrix of the latent state, so the dimension of $Q_t$
        can be smaller than the dimension of $z_t$.

The covariance matrix of the latent dynamics model takes the form $R Q R^T$, where $Q$ is
a non-singular matrix (block diagonal), and $R$ is the selecting matrix. For example,
for an STS model for a 1-d time series with a local linear component and a (dummy) seasonal
component with 4 seasons, $R$ and $Q$ takes the form
$$
Q = \begin{bmatrix}
     v_1 &  0  &  0 \\
       0  & v_2 &  0 \\
       0  &  0  & v_3
    \end{bmatrix},
\qquad
R = \begin{bmatrix}
     1 & 0 & 0 \\
     0 & 1 & 0 \\
     0 & 0 & 1 \\
     0 & 0 & 0 \\
     0 & 0 & 0
    \end{bmatrix},
$$
where $v_1$, $v_2$ are variances of the 'level' part and the 'trend' part of the
local linear component, and $v_3$ is the variance of the disturbance of the seasonal
component.



More information can be found in these books:

> -   \"Machine Learning: Advanced Topics\", K. Murphy, MIT Press 2023.
>     Available at <https://probml.github.io/pml-book/book2.html>.
> -   \"Time Series Analysis by State Space Methods (2nd edn)\", James Durbin, Siem Jan Koopman,
>     Oxford University Press, 2012.

## Example usage

The STS model is constructed by providing the observed time series and specifying a list of
components and the distribution family of the observation. The sts-jax package provides
common STS components including **local linear trend** component, **seasonal** component, 
**cycle** component, **autoregressive** component, and **regression** component. More components
will be added in the future. The observed time series can follow either the **Gaussian**
distribution or the **Poisson** distribution. 

The parameters of the STS model can be learned by **MLE**, **VI**, or **HMC**.

Below we illustrate the high level (object-oriented) API for the case of an STS with Gaussian
observations. (See [this
notebook](https://github.com/probml/dynamax/blob/main/docs/notebooks/hmm/gaussian_hmm.ipynb)
for a runnable version of this code.)

## Electricity usage example

This example is adapted from the demo of sts library of the package tensorflow_probability.
<p align="center">
<img src="./sts_jax/figures/electr_obs.png" alt="drawing" style="width:600px;"/>
<p>

```python
import sts_jax.structural_time_series.sts_model as sts

# The model has two seasonal components, one autoregressive component and one regression component.
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
    obs_time_series=demand_training_data, obs_distribution='Gaussian',
    covariates=temperature_training_data)

# Perform the MLE estimation via SGD implemented in the package dynamax.
opt_param, _losses = model.fit_mle(obs_time_series,
                                   covariates=temperature_training_data,
                                   num_steps=2000)

# After the parameter is learned, forecast
forecast_means, forecasts = model.forecast(opt_param,
                                           obs_time_series,
                                           num_forecast_steps,
                                           past_covariates=temperature_training_data,
                                           forecast_covariates=temperature_predict_data)
```

<p align="center">
<img src="./sts_jax/figures/electr_forecast.png" alt="drawing" style="width:600px;"/>
<p>


## Poisson example

```python
import sts_jax.structural_time_series.sts_model as sts

trend = sts.LocalLinearTrend()
model = sts.StructuralTimeSeries([trend],
                                 obs_distribution='Poisson',
                                 obs_time_series=counts_training)

param_samples, _log_probs = model.fit_hmc(num_samples=200,
                                          obs_time_series=counts_training)

forecasts = model.forecast(param_samples, obs_time_series, num_forecast_steps)[1]
```

<p align="center">
<img src="./sts_jax/figures/poisson_forecast.png" alt="drawing" style="width:600px;"/>
<p>

<p align="center">
<img src="./sts_jax/figures/comparison.png" alt="drawing" style="width:600px;"/>
<p>

## Causal Impact example

<p align="center">
<img src="./sts_jax/figures/causal_obs.png" alt="drawing" style="width:600px;"/>
<p>

```python
from sts_jax.causal_impact.causal_impact import causal_impact

# Run an anlysis
obs_time_series = jnp.expand_dims(y, 1)

impact = causal_impact(obs_time_series, intervention_timepoint, 'Gaussian', covariates,
                       sts_model=None, confidence_level=0.95, key=jr.PRNGKey(0), num_samples=200)

impact.plot()
```
<p align="center">
<img src="./sts_jax/figures/causal_forecast.png" alt="drawing" style="width:600px;"/>
<p>

```
impact.print_summary()
```
```
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

$$
{\left\lbrack \matrix{2 & 3 \cr 4 & 5} \right\rbrack} 
* \left\lbrack \matrix{1 & 0 \cr 0 & 1} \right\rbrack
= \left\lbrack \matrix{2 & 3 \cr 4 & 5} \right\rbrack
$$    

MIT License. 2022

https://github.com/xinglong-li/sts-jax/blob/4b08e7bf4f1bdd940afc90784322d83f409c45d6/sts_jax/figures/causal_obs.png
