from collections import OrderedDict
from dynamax.types import PRNGKey
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from jax.tree_util import tree_map, tree_flatten
from dynamax.utils.distributions import InverseWishart as IW
from dynamax.parameters import ParameterProperties as Prop
from sts_ssm import StructuralTimeSeriesSSM
from sts_components import *
from dynamax.utils.bijectors import RealToPSDBijector
from typing import List, Union, Optional
import optax
from .learning import fit_hmc, fit_vi
import tensorflow_probability.substrates.jax.bijectors as tfb


class ParamsSTS(OrderedDict):
    """A :class: 'OrderdedDict' with each item being an instance of :class: 'OrderedDict'."""
    pass


class ParamPropertiesSTS(OrderedDict):
    """A :class: 'OrderdedDict' with each item being an instance of :class: 'OrderedDict',
        having the same pytree structure with 'ParamsSTS'.
    """
    pass


class StructuralTimeSeries():
    r"""The class of the Bayesian structural time series (STS) model:

    $$y_t = H_t @ z_t + \epsilon_t, \qquad  \epsilon_t \sim \mathcal{N}(0, \Sigma_t)$$
    $$z_{t+1} = F_t z_t + R_t \eta_t, \qquad eta_t \sim \mathcal{N}(0, Q_t)$$

    where

    * $H_t$: emission matrix
    * $F_t$: transition matrix of the dynamics
    * $R_t$: subset of clumns of base vector I, and is called'selection matrix'
    * $Q_t$: nonsingular covariance matrix of the latent state

    Construct a structural time series (STS) model from a list of components

    Args:
        'components' is a list of components
    """

    def __init__(
        self,
        components: List[Union[STSComponent, STSRegression]],
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        obs_distribution: str='Gaussian',
        obs_cov_prior=None,
        obs_cov_constrainer=None,
        constant_offset: bool=True,
        name: str='sts_model'
    ) -> None:
        names = [c.name for c in components]
        assert len(set(names)) == len(names), "Components should not share the same name."
        assert obs_distribution in ['Gaussian', 'Poisson'],\
            "The distribution of observations must be Gaussian or Poisson."

        self.name = name
        self.dim_obs = obs_time_series.shape[-1]
        self.obs_distribution = obs_distribution
        self.dim_covariate = covariates.shape[-1] if covariates is not None else 0

        obs_unconstrained = self._unconstrain_obs(obs_time_series)
        self.offset = obs_unconstrained.mean(axis=0) if constant_offset else 0.
        obs_centered_unconstrained = obs_unconstrained - self.offset

        # Initialize paramters using the scale of observed time series
        regression = None
        obs_scale = jnp.std(jnp.abs(jnp.diff(obs_centered_unconstrained, axis=0)), axis=0)
        # obs_scale = jnp.std(obs_centered_unconstrained, axis=0)
        for c in components:
            if isinstance(c, STSRegression):
                regression = c
                regression.initialize_params(covariates, obs_centered_unconstrained)
                residuals = regression.get_reg_value(regression.params, covariates)
                obs_scale = jnp.std(jnp.abs(jnp.diff(residuals, axis=0)), axis=0)
        for c in components:
            if not isinstance(c, STSRegression):
                c.initialize_params(obs_centered_unconstrained[0], obs_scale)

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

        if self.obs_distribution == 'Gaussian':
            if obs_cov_prior is None:
                obs_cov_prior = IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs))
            if obs_cov_constrainer is None:
                obs_cov_constrainer = RealToPSDBijector()
            obs_cov_props = Prop(trainable=True, constrainer=obs_cov_constrainer)
            obs_cov = obs_cov_prior.mode()
            self.param_props['obs_model'] = OrderedDict({'cov': obs_cov_props})
            self.param_priors['obs_model'] = OrderedDict({'cov': obs_cov_prior})
            self.params['obs_model'] = OrderedDict({'cov': obs_cov})

        # Always put the regression term at the last position of the OrderedDict.
        if regression is not None:
            self.param_props[regression.name] = regression.param_props
            self.param_priors[regression.name] = regression.param_priors
            self.params[regression.name] = regression.params
            self.reg_func = regression.get_reg_value
        else:
            self.reg_func = None

    def as_ssm(self):
        """Formulate the STS model as a state space model.
        """
        sts_ssm = StructuralTimeSeriesSSM(self.params,
                                          self.param_props,
                                          self.param_priors,
                                          self.trans_mat_getters,
                                          self.trans_cov_getters,
                                          self.obs_mats,
                                          self.cov_select_mats,
                                          self.initial_distributions,
                                          self.reg_func,
                                          self.obs_distribution,
                                          self.dim_covariate)
        return sts_ssm

    def sample(
        self,
        num_timesteps: int,
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        key: PRNGKey=jr.PRNGKey(0),
    ):
        """Given parameters, sample latent states and corresponding observed time series.
        """
        sts_ssm = self.as_ssm()
        states, timeseries = sts_ssm.sample(key, num_timesteps, covariates)
        return self.offset + timeseries

    def marginal_log_prob(
        self,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None
    ):
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()
        return sts_ssm.marginal_log_prob(sts_ssm.params, obs_centered, covariates)

    def fit_mle(
        self,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        num_steps: int=1000,
        initial_params: ParamsSTS=None,
        param_props: ParamPropertiesSTS=None,
        optimizer: optax.GradientTransformation=optax.adam(1e-1),
        key: PRNGKey=jr.PRNGKey(0)
    ):
        """Maximum likelihood estimate of parameters of the STS model
        """
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()
        curr_params = sts_ssm.params if initial_params is None else initial_params
        if param_props is None:
            param_props = sts_ssm.param_props

        optimal_params, losses = sts_ssm.fit_sgd(
            curr_params, param_props, obs_centered, num_epochs=num_steps,
            key=key, inputs=covariates, optimizer=optimizer)

        return optimal_params, losses

    def fit_vi(
        self,
        num_samples: int,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        initial_params: ParamsSTS=None,
        param_props: ParamPropertiesSTS=None,
        num_step_iters: int=50,
        verbose: bool=True,
        key: PRNGKey=jr.PRNGKey(0)
    ):
        """Sample parameters of the STS model from ADVI posterior.

        Parameters of the STS model includes:
            covariance matrix of each component,
            regression coefficient matrix (if the model has inputs and a regression component)
            covariance matrix of observations (if observations follow Gaussian distribution)
        """
        sts_ssm = self.as_ssm()
        # Initialize via fit MLE if initial params is not given.
        if initial_params is None:
            initial_params = sts_ssm.params
        if param_props is None:
            param_props = self.param_props

        obs_centered = self._center_obs(obs_time_series)
        param_samps, losses = fit_vi(
            sts_ssm, initial_params, param_props, num_samples, obs_centered, covariates,
            key, verbose, num_step_iters, verbose)
        elbo = -losses
        return param_samps, elbo

    def fit_hmc(
        self,
        num_samples: int,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        initial_params: ParamsSTS=None,
        param_props: ParamPropertiesSTS=None,
        warmup_steps: int=100,
        verbose: bool=True,
        key: PRNGKey=jr.PRNGKey(0)
    ):
        """Sample parameters of the STS model from their posterior distributions.

        Parameters of the STS model includes:
            covariance matrix of each component,
            regression coefficient matrix (if the model has inputs and a regression component)
            covariance matrix of observations (if observations follow Gaussian distribution)
        """
        sts_ssm = self.as_ssm()
        # Initialize via fit MLE if initial params is not given.
        if initial_params is None:
            initial_params, _losses = self.fit_mle(obs_time_series, covariates, num_steps=1000)
        if param_props is None:
            param_props = self.param_props

        obs_centered = self._center_obs(obs_time_series)
        param_samps, param_log_probs = fit_hmc(
            sts_ssm, initial_params, param_props, num_samples, obs_centered, covariates,
            key, warmup_steps, verbose)
        return param_samps, param_log_probs

    def posterior_sample(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        key: PRNGKey=jr.PRNGKey(0)
    ):
        """Sample latent states given model parameters."""
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_centered = self._center_obs(obs_time_series)
        sts_ssm = self.as_ssm()
        @jit
        def single_sample(sts_param):
            # sts_ssm = StructuralTimeSeriesSSM(
            #     sts_param, self.param_props, self.param_priors,
            #     self.trans_mat_getters, self.trans_cov_getters, self.obs_mats, self.cov_select_mats,
            #     self.initial_distributions, self.reg_func, self.obs_distribution)
            ts_means, ts = sts_ssm.posterior_sample(sts_param, obs_centered, covariates, key)
            return [self._uncenter_obs(ts_means), self._uncenter_obs(ts)]

        samples = vmap(single_sample)(sts_params)

        return {'means': samples[0], 'observations': samples[1]}

    def decompose_by_component(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        num_pos_samples: int=100,
        key: PRNGKey=jr.PRNGKey(0)
    ):
        """Decompose the STS model into components and return the means and variances
           of the marginal posterior of each component.

           The marginal posterior of each component is obtained by averaging over
           conditional posteriors of that component using Kalman smoother conditioned
           on the sts_params. Each sts_params is a posterior sample of the STS model
           conditioned on observed_time_series.

           The marginal posterior mean and variance is computed using the formula
           E[X] = E[E[X|Y]],
           Var(Y) = E[Var(X|Y)] + Var[E[X|Y]],
           which holds for any random variables X and Y.

        Args:
            observed_time_series (_type_): _description_
            inputs (_type_, optional): _description_. Defaults to None.
            sts_params (_type_, optional): Posteriror samples of STS parameters.
                If not given, 'num_posterior_samples' of STS parameters will be
                sampled using self.fit_hmc.
            num_post_samples (int, optional): Number of posterior samples of STS
                parameters to be sampled using self.fit_hmc if sts_params=None.

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
            means = pos['pos_mean']
            covs = pos['pos_cov']
            # Use the formula: E[X] = E[E[X|Y]]
            mean_series = means.mean(axis=0)
            # Use the formula: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
            cov_series = jnp.mean(covs, axis=0)[..., 0] + jnp.var(means, axis=0)
            component_dists[c] = {'pos_mean': mean_series,
                                  'pos_cov': cov_series}

        return component_dists

    def forecast(
        self,
        sts_params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        num_forecast_steps: int,
        past_covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        forecast_covariates: Optional[Float[Array, "num_forecast_steps dim_covariates"]]=None,
        key: PRNGKey=jr.PRNGKey(0)
    ):
        """Forecast.
        """
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_centered = self._center_obs(obs_time_series)

        @jit
        def single_forecast(sts_param):
            sts_ssm = self.as_ssm()
            means, covs, ts = sts_ssm.forecast(
                sts_param, obs_centered, num_forecast_steps,
                past_covariates, forecast_covariates, key)
            return [self._uncenter_obs(means), covs, self._uncenter_obs(ts)]

        forecasts = vmap(single_forecast)(sts_params)

        return {'means': forecasts[0], 'covariances': forecasts[1], 'observations': forecasts[2]}

    def _ensure_param_has_batch_dim(self, sts_params):
        """Turn parameters into batch if only one parameter is given"""
        # All latent components except for 'LinearRegression' have transition covariances
        # and the linear regression has coefficient matrix.
        # When the observation is Gaussian, the observation model also has a covariance matrix.
        # So here we assume that the largest dimension of parameters is 2.
        param_list, _ = tree_flatten(sts_params)
        max_params_dim = max([len(x.shape) for x in param_list])
        if max_params_dim > 2:
            return sts_params
        else:
            return tree_map(lambda x: jnp.expand_dims(x, 0), sts_params)

    def _constrain_obs(self, obs_time_series):
        if self.obs_distribution == 'Gaussian':
            return obs_time_series
        elif self.obs_distribution == 'Poisson':
            return jnp.exp(obs_time_series)

    def _unconstrain_obs(self, obs_time_series_constrained):
        if self.obs_distribution == 'Gaussian':
            return obs_time_series_constrained
        elif self.obs_distribution == 'Poisson':
            return jnp.log(obs_time_series_constrained)

    def _center_obs(self, obs_time_series):
        obs_unconstrained = self._unconstrain_obs(obs_time_series)
        return self._constrain_obs(obs_unconstrained - self.offset)

    def _uncenter_obs(self, obs_time_series_centered):
        obs_centered_unconstrained = self._unconstrain_obs(obs_time_series_centered)
        return self._constrain_obs(obs_centered_unconstrained + self.offset)
