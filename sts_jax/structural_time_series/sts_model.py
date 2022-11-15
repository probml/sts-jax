from collections import OrderedDict
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from jax.tree_util import tree_map, tree_flatten
from dynamax.distributions import InverseWishart as IW
from dynamax.parameters import ParameterProperties as Prop
from dynamax.structural_time_series.models.sts_ssm import StructuralTimeSeriesSSM
from dynamax.structural_time_series.models.sts_components import *
from dynamax.utils import PSDToRealBijector
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb


RealToPSD = tfb.Invert(PSDToRealBijector)


class StructuralTimeSeries():
    """The class of the Bayesian structural time series (STS) model:

    y_t = H_t @ z_t + \err_t,   \err_t \sim N(0, \Sigma_t) 
    z_{t+1} = F_t @ z_t + R_t @ \eta_t, eta_t \sim N(0, Q_t)

    H_t: emission matrix
    F_t: transition matrix of the dynamics
    R_t: subset of clumns of base vector I, and is called'selection matrix'
    Q_t: nonsingular covariance matrix of the latent state

    Construct a structural time series (STS) model from a list of components

    Args:
        components: list of components
        observation_covariance:
        observation_covariance_prior: InverseWishart prior for the observation covariance matrix
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        name (str): name of the STS model
    """

    def __init__(self,
                 components,
                 obs_time_series,
                 obs_distribution='Gaussian',
                 obs_cov_props=None,
                 obs_cov_prior=None,
                 obs_cov=None,
                 covariates=None,
                 constant_offset=True,
                 name='sts_model'):

        names = [c.name for c in components]
        assert len(set(names)) == len(names), "Components should not share the same name."
        assert obs_distribution in ['Gaussian', 'Poisson'],\
            "The distribution of observations must be Gaussian or Poisson."

        self.name = name
        self.dim_obs = obs_time_series.shape[-1]
        self.obs_distribution = obs_distribution
        self.dim_covariate = covariates.shape[-1] if covariates is not None else 0

        self.offset = obs_time_series.mean(axis=0) if constant_offset else 0.
        obs_time_series = obs_time_series - self.offset

        # Initialize paramters using the scale of observed time series
        regression = None
        obs_scale = jnp.std(jnp.abs(jnp.diff(obs_time_series, axis=0)), axis=0)
        for c in components:
            if isinstance(c, STSRegression):
                regression = c
                regression.initialize(covariates, obs_time_series)
                residuals = regression.fit(regression.params, covariates)
                obs_scale = jnp.std(jnp.abs(jnp.diff(residuals, axis=0)), axis=0)
        for c in components:
            if not isinstance(c, STSRegression):
                c.initialize_params(obs_time_series[0], obs_scale)

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
            if obs_cov_props is None:
                obs_cov_props = Prop(trainable=True, constrainer=RealToPSD)
            if obs_cov_prior is None:
                obs_cov_prior = IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs))
            if obs_cov is None:
                obs_cov = obs_cov_prior.mode()
            self.param_props['obs_model'] = OrderedDict({'cov': obs_cov_props})
            self.param_priors['obs_model'] = OrderedDict({'cov': obs_cov_prior})
            self.params['obs_model'] = OrderedDict({'cov': obs_cov})

        # Always put the regression term at the last position of the OrderedDict.
        if regression is not None:
            self.param_props[regression.name] = regression.param_props
            self.param_priors[regression.name] = regression.param_priors
            self.params[regression.name] = regression.params
            self.reg_func = regression.fit
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

    def sample(self, key, num_timesteps, covariates=None):
        """Given parameters, sample latent states and corresponding observed time series.
        """
        sts_ssm = self.as_ssm()
        states, timeseries = sts_ssm.sample(key, num_timesteps, covariates)
        return self.offset + timeseries

    def marginal_log_prob(self, obs_time_series, covariates=None):
        obs_time_series = obs_time_series - self.offset
        sts_ssm = self.as_ssm()
        return sts_ssm.marginal_log_prob(sts_ssm.params, obs_time_series, covariates)

    def fit_mle(self,
                obs_time_series,
                covariates=None,
                num_steps=1000,
                initial_params=None,
                optimizer=optax.adam(1e-1),
                key=jr.PRNGKey(0)):
        """Maximum likelihood estimate of parameters of the STS model
        """
        obs_time_series = obs_time_series - self.offset
        sts_ssm = self.as_ssm()
        curr_params = sts_ssm.params if initial_params is None else initial_params
        param_props = sts_ssm.param_props

        optimal_params, losses = sts_ssm.fit_sgd(
            curr_params, param_props, obs_time_series, num_epochs=num_steps,
            key=key, inputs=covariates, optimizer=optimizer)

        return optimal_params, losses

    def fit_hmc(self,
                num_samples,
                obs_time_series,
                covariates=None,
                initial_params=None,
                param_props=None,
                warmup_steps=100,
                verbose=True,
                key=jr.PRNGKey(0)):
        """Sample parameters of the STS model from their posterior distributions.

        Parameters of the STS model includes:
            covariance matrix of each component,
            regression coefficient matrix (if the model has inputs and a regression component)
            covariance matrix of observations (if observations follow Gaussian distribution)
        """
        obs_time_series = obs_time_series - self.offset
        sts_ssm = self.as_ssm()
        # Initialize via fit MLE if initial params is not given.
        if initial_params is None:
            initial_params, _losses = self.fit_mle(obs_time_series, covariates, num_steps=1000)
        if param_props is None:
            param_props = self.param_props

        param_samps, param_log_probs = sts_ssm.fit_hmc(
            initial_params, param_props, key, num_samples, obs_time_series, covariates,
            warmup_steps, verbose)
        return param_samps, param_log_probs

    def posterior_sample(self, key, obs_time_series, sts_params, inputs=None):
        """Sample latent states given model parameters."""
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_time_series = obs_time_series - self.offset

        @jit
        def single_sample(sts_param):
            sts_ssm = StructuralTimeSeriesSSM(
                sts_param, self.param_props, self.param_priors,
                self.trans_mat_getters, self.trans_cov_getters, self.obs_mats, self.cov_select_mats,
                self.initial_distributions, self.reg_func, self.obs_distribution)
            ts_means, ts = sts_ssm.posterior_sample(key, obs_time_series, inputs)
            return [self.offset + ts_means, self.offset + ts]

        samples = vmap(single_sample)(sts_params)

        return {'means': samples[0], 'observations': samples[1]}

    def decompose_by_component(self, sts_params, obs_time_series, covariates=None,
                               num_pos_samples=100, key=jr.PRNGKey(0)):
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
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_time_series = obs_time_series - self.offset

        component_dists = OrderedDict()

        # Sample parameters from the posterior if parameters is not given
        if sts_params is None:
            sts_params = self.fit_hmc(num_pos_samples, obs_time_series, covariates, key=key)

        @jit
        def single_decompose(sts_param):
            sts_ssm = self.as_ssm()
            return sts_ssm.component_posterior(sts_param, obs_time_series, covariates)

        component_conditional_pos = vmap(single_decompose)(sts_params)

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

    def forecast(self,
                 sts_params,
                 obs_time_series,
                 num_forecast_steps,
                 past_covariates=None,
                 forecast_covariates=None,
                 key=jr.PRNGKey(0)):
        """Forecast.
        """
        sts_params = self._ensure_param_has_batch_dim(sts_params)
        obs_time_series = obs_time_series - self.offset

        @jit
        def single_forecast(sts_param):
            sts_ssm = self.as_ssm()
            means, covs, ts = sts_ssm.forecast(
                sts_param, obs_time_series, num_forecast_steps,
                past_covariates, forecast_covariates, key)
            return [means + self.offset, covs, ts + self.offset]

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
