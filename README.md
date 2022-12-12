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