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
> -   \"Bayesian Filtering and Smoothing\", S. Särkkä, Cambridge
>     University Press, 2013. Available at
>     <https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf>

## Example usage

Dynamax includes classes for many kinds of SSM. You can use these models
to simulate data, and you can fit the models using standard learning
algorithms like expectation-maximization (EM) and stochastic gradient
descent (SGD). Below we illustrate the high level (object-oriented) API
for the case of an HMM with Gaussian emissions. (See [this
notebook](https://github.com/probml/dynamax/blob/main/docs/notebooks/hmm/gaussian_hmm.ipynb)
for a runnable version of this code.)

## CO2 example
```python
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from dynamax.hidden_markov_model import GaussianHMM

key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
num_states = 3
emission_dim = 2
num_timesteps = 1000

# Make a Gaussian HMM and sample data from it
hmm = GaussianHMM(num_states, emission_dim)
true_params, _ = hmm.initialize(key1)
true_states, emissions = hmm.sample(true_params, key2, num_timesteps)

# Make a new Gaussian HMM and fit it with EM
params, props = hmm.initialize(key3, method="kmeans", emissions=emissions)
params, lls = hmm.fit_em(params, props, emissions, num_iters=20)

# Plot the marginal log probs across EM iterations
plt.plot(lls)
plt.xlabel("EM iterations")
plt.ylabel("marginal log prob.")

# Use fitted model for posterior inference
post = hmm.smoother(params, emissions)
print(post.smoothed_probs.shape) # (1000, 3)
```

JAX allows you to easily vectorize these operations with `vmap`.
For example, you can sample and fit to a batch of emissions as shown below.

```python
from functools import partial
from jax import vmap

num_seq = 200
batch_true_states, batch_emissions = \
    vmap(partial(hmm.sample, true_params, num_timesteps=num_timesteps))(
        jr.split(key2, num_seq))
print(batch_true_states.shape, batch_emissions.shape) # (200,1000) and (200,1000,2)

# Make a new Gaussian HMM and fit it with EM
params, props = hmm.initialize(key3, method="kmeans", emissions=batch_emissions)
params, lls = hmm.fit_em(params, props, batch_emissions, num_iters=20)
```

These examples demonstrate the dynamax models, but we can also call the low-level
inference code directly.

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/sts_jax/main/docs/figures/electr_obs.png">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/sts_jax/main/docs/figures/electr_forecast.png">
</p>

## Poisson example
```python
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from dynamax.hidden_markov_model import GaussianHMM

key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
num_states = 3
emission_dim = 2
num_timesteps = 1000

# Make a Gaussian HMM and sample data from it
hmm = GaussianHMM(num_states, emission_dim)
true_params, _ = hmm.initialize(key1)
true_states, emissions = hmm.sample(true_params, key2, num_timesteps)

# Make a new Gaussian HMM and fit it with EM
params, props = hmm.initialize(key3, method="kmeans", emissions=emissions)
params, lls = hmm.fit_em(params, props, emissions, num_iters=20)

# Plot the marginal log probs across EM iterations
plt.plot(lls)
plt.xlabel("EM iterations")
plt.ylabel("marginal log prob.")

# Use fitted model for posterior inference
post = hmm.smoother(params, emissions)
print(post.smoothed_probs.shape) # (1000, 3)
```

JAX allows you to easily vectorize these operations with `vmap`.
For example, you can sample and fit to a batch of emissions as shown below.

```python
from functools import partial
from jax import vmap

num_seq = 200
batch_true_states, batch_emissions = \
    vmap(partial(hmm.sample, true_params, num_timesteps=num_timesteps))(
        jr.split(key2, num_seq))
print(batch_true_states.shape, batch_emissions.shape) # (200,1000) and (200,1000,2)

# Make a new Gaussian HMM and fit it with EM
params, props = hmm.initialize(key3, method="kmeans", emissions=batch_emissions)
params, lls = hmm.fit_em(params, props, batch_emissions, num_iters=20)
```

These examples demonstrate the dynamax models, but we can also call the low-level
inference code directly.

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/sts_jax/main/docs/figures/poisson_obs.png">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/sts_jax/main/docs/figures/poisson_forecast.png">
</p>

## Causal Impact example
```python
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from dynamax.hidden_markov_model import GaussianHMM

key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
num_states = 3
emission_dim = 2
num_timesteps = 1000

# Make a Gaussian HMM and sample data from it
hmm = GaussianHMM(num_states, emission_dim)
true_params, _ = hmm.initialize(key1)
true_states, emissions = hmm.sample(true_params, key2, num_timesteps)

# Make a new Gaussian HMM and fit it with EM
params, props = hmm.initialize(key3, method="kmeans", emissions=emissions)
params, lls = hmm.fit_em(params, props, emissions, num_iters=20)

# Plot the marginal log probs across EM iterations
plt.plot(lls)
plt.xlabel("EM iterations")
plt.ylabel("marginal log prob.")

# Use fitted model for posterior inference
post = hmm.smoother(params, emissions)
print(post.smoothed_probs.shape) # (1000, 3)
```

JAX allows you to easily vectorize these operations with `vmap`.
For example, you can sample and fit to a batch of emissions as shown below.

```python
from functools import partial
from jax import vmap

num_seq = 200
batch_true_states, batch_emissions = \
    vmap(partial(hmm.sample, true_params, num_timesteps=num_timesteps))(
        jr.split(key2, num_seq))
print(batch_true_states.shape, batch_emissions.shape) # (200,1000) and (200,1000,2)

# Make a new Gaussian HMM and fit it with EM
params, props = hmm.initialize(key3, method="kmeans", emissions=batch_emissions)
params, lls = hmm.fit_em(params, props, batch_emissions, num_iters=20)
```

These examples demonstrate the dynamax models, but we can also call the low-level
inference code directly.

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/sts_jax/main/docs/figures/causal_obs.png">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/sts_jax/main/docs/figures/causal_forecast.png">
</p>

## Contributing

Please see [this page](https://github.com/probml/dynamax/blob/main/CONTRIBUTING.md) for details
on how to contribute.

## About

MIT License. 2022