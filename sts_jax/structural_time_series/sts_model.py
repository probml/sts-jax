from collections import OrderedDict
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.parameters import ParameterProperties
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import InverseWishart as IW
from jax import jit, vmap
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax import distributions as tfd

from .learning import fit_hmc, fit_vi
from .sts_components import *
from .sts_ssm import StructuralTimeSeriesSSM


class StructuralTimeSeries:
    r"""The class of the Bayesian structural time series (STS) model.

    The STS model is a linear state space model with a specific structure. In particular,
    the latent state $z_t$ is a composition of states of all latent components:

    $$z_t = [c_{1, t}, c_{2, t}, ...]$$

    where $c_{i,t}$ is the state of latent component $c_i$ at time step $t$.

    The STS model takes the form:

    $$y_t = H_t z_t + u_t + \epsilon_t, \qquad  \epsilon_t \sim \mathcal{N}(0, \Sigma_t)$$
    $$z_{t+1} = F_t z_t + R_t \eta_t, \qquad eta_t \sim \mathcal{N}(0, Q_t)$$

    where

    * $H_t$: emission matrix, which sums up the contributions of all latent components.
    * $u_t$: is the contribution of the regression component.
    * $F_t$: transition matrix of the latent dynamics
    * $R_t$: the selection matrix, which is a subset of clumnes of base vector $I$, converting
        the non-singular covariance matrix into a (possibly singular) covariance matrix for
        the latent state $z_t$.
    * $Q_t$: nonsingular covariance matrix of the latent state, so the dimension of $Q_t$
        can be smaller than the dimension of $z_t$.

    The covariance matrix of the latent dynamics model takes the form $R Q R^T$, where $Q$ is
    a nonsingular matrix (block diagonal), and $R$ is the selecting matrix. For example,
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
        \end{bmatrix}
    $$

    where $v_1$, $v_2$ are variances of the 'level' part and the 'trend' part of the
    local linear component, and $v_3$ is the variance of the disturbance of the seasonal
    component.
    """

    def __init__(
        self,
        components: List[Union[STSComponent, STSRegression]],
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        obs_distribution: str = "Gaussian",
        obs_cov_prior: tfd.Distribution = None,
        obs_cov_constrainer: tfb.Bijector = None,
        constant_offset: bool = True,
        name: str = "sts_model",
    ) -> None:
        r"""
        Args:
            components: list of components of the STS model, could be instances of
                STSComponent and STSRegression, each component must have a unique name.
            obs_time_series: observed time series to be modeled.
            covariates: series of the covariates (if there is a STSRegression component).
            obs_distribution: the distribution family of the observed time series.
                Currently it can be either 'Gaussian' or 'Poisson'.
            obs_cov_prior: the prior distribution of the covariance matrix of the observation
                $y_t$ at each time step. This is only required when obs_distribution='Gaussian'.
            obs_cov_constrainer: a bijector whose inverse operator transforms the observation
                covariance matrix to unconstrained space, for the purpose of learning.
            constant_offset: If true, the observed time series will be centered before it is
                fitted. If obs_distribution='Poisson', log(obs_time_series) is centered.
            name: name of the STS model.
        """
        names = [c.name for c in components]
        assert len(set(names)) == len(names), "Components should not share the same name."
        assert obs_distribution in [
            "Gaussian",
            "Poisson",
        ], "The distribution of observations must be Gaussian or Poisson."

        self.name = name
        self.dim_obs = obs_time_series.shape[-1]
        self.obs_distribution = obs_distribution
        self.dim_covariate = covariates.shape[-1] if covariates is not None else 0

        # Convert the time series into the unconstrained space if obs_distribution is not Gaussian
        obs_unconstrained = self._unconstrain_obs(obs_time_series)
        self.offset = obs_unconstrained.mean(axis=0) if constant_offset else 0.0
        obs_centered = self._center_obs(obs_time_series)
        obs_centered_unconstrained = self._unconstrain_obs(obs_centered)

        # Initialize model paramters with the observed time series
        initial = obs_centered_unconstrained[0]
        obs_scale = jnp.std(jnp.abs(jnp.diff(obs_centered_unconstrained, axis=0)), axis=0)
        # If a regression component is included, remove the effect of the regression model
        # before initializing parameters of the time series.
        regression = None
        for c in components:
            if isinstance(c, STSRegression):
                assert len(components) > 1, "The STS model cannot only contain one regresion component!"
                regression = c
                regression.initialize_params(covariates, obs_centered_unconstrained)
                residuals = obs_centered_unconstrained - regression.get_reg_value(regression.params, covariates)
                initial = residuals[0]
                obs_scale = jnp.std(jnp.abs(jnp.diff(residuals, axis=0)), axis=0)
        for c in components:
            if not isinstance(c, STSRegression):
                c.initialize_params(initial, obs_scale)

        # Aggeragate components
        self.initial_distributions = OrderedDict()
        self.param_props = OrderedDict()
        self.param_priors = OrderedDict()
        self.params = OrderedDict()
        self.trans_mat_getters = OrderedDict()
        self.trans_cov_getters = OrderedDict()
        self.obs_mats = OrderedDict()
        self.cov_select_mats = OrderedDict()

        for c in components:
            if not isinstance(c, STSRegression):
                self.initial_distributions[c.name] = c.initial_distribution
                self.param_props[c.name] = c.param_props
                self.param_priors[c.name] = c.param_priors
                self.params[c.name] = c.params
                self.trans_mat_getters[c.name] = c.get_trans_mat
                self.trans_cov_getters[c.name] = c.get_trans_cov
                self.obs_mats[c.name] = c.obs_mat
                self.cov_select_mats[c.name] = c.cov_select_mat

        # Add parameters of the observation model if the observed time series is
        # normally distributed.
        if self.obs_distribution == "Gaussian":
            if obs_cov_prior is None:
                obs_cov_prior = IW(df=self.dim_obs, scale=1e-4 * obs_scale**2 * jnp.eye(self.dim_obs))
            if obs_cov_constrainer is None:
                obs_cov_constrainer = RealToPSDBijector()
            obs_cov_props = ParameterProperties(trainable=True, constrainer=obs_cov_constrainer)
            obs_cov = obs_cov_prior.mode()
            self.param_props["obs_model"] = OrderedDict({"cov": obs_cov_props})
            self.param_priors["obs_model"] = OrderedDict({"cov": obs_cov_prior})
            self.params["obs_model"] = OrderedDict({"cov": obs_cov})

        # Always put the regression term at the last position of the OrderedDict.
        if regression is not None:
            self.param_props[regression.name] = regression.param_props
            self.param_priors[regression.name] = regression.param_priors
            self.params[regression.name] = regression.params
            self.reg_func = regression.get_reg_value
        else:
            self.reg_func = None

    def as_ssm(self) -> StructuralTimeSeriesSSM:
        """Convert the STS model as a state space model."""
        sts_ssm = StructuralTimeSeriesSSM(
            self.params,
            self.param_props,
            self.param_priors,
            self.trans_mat_getters,
            self.trans_cov_getters,
            self.obs_mats,
            self.cov_select_mats,
            self.initial_distributions,
            self.reg_func,
            self.obs_distribution,
            self.dim_covariate,
        )
        return sts_ssm

    def sample(
        self,
        sts_params: ParamsSTS,
        num_timesteps: int,
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[Float[Array, "num_timesteps dim_obs"], Float[Array, "num_timesteps dim_obs"]]:
        """Sample observed time series given model parameters."""
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        sts_ssm = self.as_ssm()

        @jit
        def single_sample(sts_param):
            sample_mean, sample_obs = sts_ssm.sample(sts_param, num_timesteps, covariates, key)
            return self._uncenter_obs(sample_mean), self._uncenter_obs(sample_obs)

        sample_means, sample_obs = vmap(single_sample)(sts_params)

        return sample_means, sample_obs

    def marginal_log_prob(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
    ) -> Scalar:
        """Compute marginal log likelihood of the observed time series given model parameters."""
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()

        return sts_ssm.marginal_log_prob(sts_params, obs_centered, covariates)

    def posterior_sample(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[Float[Array, "num_params num_timesteps dim_obs"], Float[Array, "num_params num_timesteps dim_obs"]]:
        """Sample latent states from their posterior given model parameters."""
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()

        @jit
        def single_sample(sts_param):
            predictive_mean, predictive_obs = sts_ssm.posterior_sample(sts_param, obs_centered, covariates, key)
            return self._uncenter_obs(predictive_mean), self._uncenter_obs(predictive_obs)

        predictive_means, predictive_samples = vmap(single_sample)(sts_params)

        return predictive_means, predictive_samples

    def fit_mle(
        self,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        num_steps: int = 1000,
        initial_params: ParamsSTS = None,
        param_props: ParamPropertiesSTS = None,
        optimizer: optax.GradientTransformation = optax.adam(1e-1),
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[ParamsSTS, Float[Array, "num_steps"]]:
        """Perform maximum likelihood estimate of parameters of the STS model."""
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()
        curr_params = sts_ssm.params if initial_params is None else initial_params
        if param_props is None:
            param_props = sts_ssm.param_props

        optimal_params, losses = sts_ssm.fit_sgd(
            curr_params,
            param_props,
            obs_centered,
            num_epochs=num_steps,
            key=key,
            inputs=covariates,
            optimizer=optimizer,
        )

        return optimal_params, losses

    def fit_vi(
        self,
        num_samples: int,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        initial_params: ParamsSTS = None,
        param_props: ParamPropertiesSTS = None,
        num_step_iters: int = 50,
        key: PRNGKey = jr.PRNGKey(0),
    ):
        """Sample parameters of the STS model from ADVI posterior."""
        sts_ssm = self.as_ssm()
        if initial_params is None:
            initial_params = sts_ssm.params
        if param_props is None:
            param_props = sts_ssm.param_props

        obs_centered = self._center_obs(obs_time_series)
        param_samps, losses = fit_vi(
            sts_ssm, initial_params, param_props, num_samples, obs_centered, covariates, key, num_step_iters
        )
        elbo = -losses

        return param_samps, elbo

    def fit_hmc(
        self,
        num_samples: int,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        initial_params: ParamsSTS = None,
        param_props: ParamPropertiesSTS = None,
        warmup_steps: int = 100,
        verbose: bool = True,
        key: PRNGKey = jr.PRNGKey(0),
    ):
        """Sample parameters of the STS model from their posterior distributions with HMC (NUTS)."""
        sts_ssm = self.as_ssm()
        # Initialize via fit MLE if initial params is not given.
        if initial_params is None:
            initial_params, _losses = self.fit_mle(obs_time_series, covariates, num_steps=500)
        if param_props is None:
            param_props = self.param_props

        obs_centered = self._center_obs(obs_time_series)
        param_samps, param_log_probs = fit_hmc(
            sts_ssm, initial_params, param_props, num_samples, obs_centered, covariates, key, warmup_steps, verbose
        )
        return param_samps, param_log_probs

    def decompose_by_component(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        num_pos_samples: int = 100,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> OrderedDict:
        r"""Decompose the STS model into components and return the means and variances
           of the marginal posterior of each component.

           The marginal posterior of each component is obtained by averaging over
           conditional posteriors of that component using Kalman smoother conditioned
           on the sts_params. Each sts_params is a posterior sample of the STS model
           conditioned on observed_time_series.

           The marginal posterior mean and variance is computed using the formula
           $$E[X] = E[E[X|Y]]$$,
           $$Var(Y) = E[Var(X|Y)] + Var[E[X|Y]]$$
           which holds for any random variables X and Y.

        Returns:
            component_dists: (OrderedDict) each item is a tuple of means and variances
                              of one component.
        """

        # Sample parameters from the posterior if parameters is not given
        if sts_params is None:
            sts_params = self.fit_hmc(num_pos_samples, obs_time_series, covariates, key=key)

        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_centered = self._center_obs(obs_time_series)

        @jit
        def single_decompose(sts_param):
            sts_ssm = self.as_ssm()
            return sts_ssm.component_posterior(sts_param, obs_centered, covariates)

        component_conditional_pos = vmap(single_decompose)(sts_params)

        component_dists = OrderedDict()
        # Obtain the marginal posterior
        for c, pos in component_conditional_pos.items():
            means = pos["pos_mean"]
            covs = pos["pos_cov"]
            # Use the formula: E[X] = E[E[X|Y]]
            mean_series = means.mean(axis=0)
            # Use the formula: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
            cov_series = jnp.mean(covs, axis=0)[..., 0] + jnp.var(means, axis=0)
            component_dists[c] = {"pos_mean": mean_series, "pos_cov": cov_series}

        return component_dists

    def forecast(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        num_forecast_steps: int,
        num_forecast_samples: int = 100,
        past_covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        forecast_covariates: Optional[Float[Array, "num_forecast_steps dim_covariates"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[
        Float[Array, "num_params num_forecast_samples num_forecast_steps dim_obs"],
        Float[Array, "num_params num_forecast_samples num_forecast_steps dim_obs"],
    ]:
        """Forecast.

        Args:
            sts_params: parameters of the STS model, or a batch of STS parameters.
            obs_time_series: observed time series.
            num_forecast_steps: number of steps of forecast.
            num_forecast_samples: number of samples for each STS parameter, used to compute
                summary statistics of the forecast, conditioned on the STS parameter.
            past_covariates: inputs of the regression component of the STS model, corresponding
                to the observed time series.
            forecast_covariates: inputs of the regression component of the STS model,
                used in forecasting.
        """
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()

        @jit
        def single_forecast(sts_param):
            _forecast_mean, _forecast_obs = sts_ssm.forecast(
                sts_param,
                obs_centered,
                num_forecast_steps,
                num_forecast_samples,
                past_covariates,
                forecast_covariates,
                key,
            )
            forecast_mean = vmap(self._uncenter_obs)(_forecast_mean)
            forecast_obs = vmap(self._uncenter_obs)(_forecast_obs)
            return forecast_mean, forecast_obs

        forecast_means, forecast_obs = vmap(single_forecast)(sts_params)

        return forecast_means, forecast_obs

    def _ensure_param_has_batch_dim(self, sts_params):
        """Turn parameters into batch if only one parameter is given"""
        # All latent components except for 'LinearRegression' have transition covariances
        # and the linear regression has coefficient matrix.
        # When the observation is Gaussian, the observation model also has a covariance matrix.
        # So here we assume that the largest dimension of parameters is 2.
        param_list = tree_leaves(sts_params)
        max_params_dim = max([len(x.shape) for x in param_list])
        if max_params_dim > 2:
            return sts_params
        else:
            return tree_map(lambda x: jnp.expand_dims(x, 0), sts_params)

    def _constrain_obs(self, obs_time_series):
        if self.obs_distribution == "Gaussian":
            return obs_time_series
        elif self.obs_distribution == "Poisson":
            return jnp.exp(obs_time_series)

    def _unconstrain_obs(self, obs_time_series_constrained):
        if self.obs_distribution == "Gaussian":
            return obs_time_series_constrained
        elif self.obs_distribution == "Poisson":
            return jnp.log(obs_time_series_constrained)

    def _center_obs(self, obs_time_series):
        if self.obs_distribution == "Gaussian":
            obs_unconstrained = self._unconstrain_obs(obs_time_series)
            return self._constrain_obs(obs_unconstrained - self.offset)
        elif self.obs_distribution == "Poisson":
            return obs_time_series

    def _uncenter_obs(self, obs_time_series_centered):
        if self.obs_distribution == "Gaussian":
            obs_centered_unconstrained = self._unconstrain_obs(obs_time_series_centered)
            return self._constrain_obs(obs_centered_unconstrained + self.offset)
        elif self.obs_distribution == "Poisson":
            return obs_time_series_centered
