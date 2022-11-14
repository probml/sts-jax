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


**Note: this is still a work in progress, and is not yet fully compatible with the latest version of dynamax.**
