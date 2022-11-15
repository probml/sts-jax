from jax import lax, vmap
from collections import OrderedDict
from functools import partial
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from dynamax.ssm import SSM
from dynamax.generalized_gaussian_ssm import (
    ParamsGGSSM,
    EKFIntegrals,
    iterated_conditional_moments_gaussian_filter as cmgf_filt,
    iterated_conditional_moments_gaussian_smoother as cmgf_smooth,
    EKFIntegrals)
from dynamax.linear_gaussian_ssm import (
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    lgssm_filter,
    lgssm_smoother,
    lgssm_posterior_sample)
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    Poisson as Pois)


class StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.

    The covariance matrix of the latent dynamics model takes the form:
    R @ Q, where Q is a dense matrix (blockwise diagonal),
    and R is the sparsing matrix. For example,
    for an STS model for a 1-d time series with a local linear component
    and a seasonal component with 4 seasons:
                                        | 1, 0, 0 |
                | v1,   0,  0 |         | 0, 1, 0 |
            Q = |  0,  v2,  0 |,    R = | 0, 0, 1 |
                |  0,   0, v3 |         | 0, 0, 0 |
                                        | 0, 0, 0 |
    """

    def __init__(self,
                 params,
                 param_props,
                 param_priors,
                 trans_mat_getters,
                 trans_cov_getters,
                 obs_mats,
                 cov_select_mats,
                 initial_distributions,
                 reg_func=None,
                 obs_distribution='Gaussian',
                 dim_covariate=0):
        self.params = params
        self.param_props = param_props
        self.param_priors = param_priors

        self.trans_mat_getters = trans_mat_getters
        self.trans_cov_getters = trans_cov_getters
        self.component_obs_mats = obs_mats
        self.cov_select_mats = cov_select_mats

        self.latent_comp_names = cov_select_mats.keys()
        self.obs_distribution = obs_distribution

        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in initial_distributions.values()])
        self.initial_cov = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in initial_distributions.values()])

        self.obs_mat = jnp.concatenate([*obs_mats.values()], axis=1)
        self.cov_select_mat = jsp.linalg.block_diag(*cov_select_mats.values())

        # Dimensions of the SSM.
        self.dim_obs, self.dim_state = self.obs_mat.shape
        self.dim_comp = self.get_trans_cov(self.params, 0).shape[0]
        self.dim_covariate = dim_covariate

        # Pick out the regression component if there is one.
        if reg_func is not None:
            # The regression component is always the last component if there is one.
            self.reg_name = list(params.keys())[-1]
            self.regression = reg_func

    @property
    def emission_shape(self):
        return (self.dim_obs,)

    @property
    def inputs_shape(self):
        return (self.dim_covariate,)

    def log_prior(self, params):
        """Log prior probability of parameters.
        """
        lp = 0.
        for c_name, c_priors in self.param_priors.items():
            for p_name, p_pri in c_priors.items():
                lp += p_pri.log_prob(params[c_name][p_name])
        return lp

    # Instantiate distributions of the SSM model
    def initial_distribution(self):
        """Distribution of the initial state of the SSM form of the STS model.
        """
        return MVN(self.initial_mean, self.initial_cov)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
           multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, inputs):
        """Depends on the distribution family of the observation.
        """
        if self.obs_distribution == 'Gaussian':
            return MVN(self.obs_mat @ state + inputs, self.params['obs_model']['cov'])
        elif self.obs_distribution == 'Poisson':
            unc_rates = self.obs_mat @ state + inputs
            return Pois(rate=self._emission_constrainer(unc_rates))

    def sample(self, params, num_timesteps, covariates=None, key=jr.PRNGKey(0)):
        """Sample a sequence of latent states and emissions with the given parameters.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros((num_timesteps, self.dim_obs))

        get_trans_mat = partial(self.get_trans_mat, params)
        get_trans_cov = partial(self.get_trans_cov, params)

        def _step(prev_state, args):
            key, input, t = args
            key1, key2 = jr.split(key, 2)
            _next_state = get_trans_mat(t) @ prev_state
            next_state = _next_state + self.cov_select_mat @ MVN(
                jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)
            prev_obs = self.emission_distribution(prev_state, input).sample(seed=key2)
            return next_state, (prev_state, prev_obs)

        # Sample the initial state.
        key1, key2 = jr.split(key, 2)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the following emissions and states.
        key2s = jr.split(key2, num_timesteps)
        _, (states, time_series) = lax.scan(
            _step, initial_state, (key2s, inputs, jnp.arange(num_timesteps)))
        return states, time_series

    def marginal_log_prob(self, params, obs_time_series, covariates=None):
        """Compute log marginal likelihood of observations.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the model to SSM.
        ssm_params = self._to_ssm_params(params)

        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=inputs)
        return filtered_posterior.marginal_loglik

    def posterior_sample(self, params, obs_time_series, covariates=None, key=jr.PRNGKey(0)):
        """Posterior sample.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)

        # Sample latent state.
        key1, key2 = jr.split(key, 2)
        ll, states = self._ssm_posterior_sample(ssm_params, obs_time_series, inputs, key1)

        # Sample observations.
        unc_obs_means = states @ self.obs_mat.T + inputs
        obs_means = self._emission_constrainer(unc_obs_means)
        key2s = jr.split(key2, obs_time_series.shape[0])
        obs_sampler = lambda state, input, key:\
            self.emission_distribution(state, input).sample(seed=key)
        obs = vmap(obs_sampler)(states, inputs, key2s)
        return obs_means, obs

    def component_posterior(self, params, obs_time_series, covariates=None):
        """Smoothing by component.
        """
        component_pos = OrderedDict()
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
            component_pos[self.reg_name] = {'pos_mean': jnp.squeeze(inputs),
                                            'pos_cov': jnp.zeros_like(obs_time_series)}
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)

        # Infer the posterior of the joint SSM model.
        pos = self._ssm_smoother(ssm_params, obs_time_series, inputs)
        mu_pos = pos.smoothed_means
        var_pos = pos.smoothed_covariances

        # Decompose by latent component.
        _loc = 0
        for c, obs_mat in self.component_obs_mats.items():
            # Extract posterior mean and covariances of each component
            # from the joint latent state.
            c_dim = obs_mat.shape[1]
            c_mean = mu_pos[:, _loc:_loc+c_dim]
            c_cov = var_pos[:, _loc:_loc+c_dim, _loc:_loc+c_dim]
            # Posterior emission of the single component.
            c_obs_mean = vmap(jnp.matmul, (None, 0))(obs_mat, c_mean)
            c_obs_constrained_mean = self._emission_constrainer(c_obs_mean)
            c_obs_cov = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_cov, obs_mat)
            component_pos[c] = {'pos_mean': c_obs_constrained_mean,
                                'pos_cov': c_obs_cov}
            _loc += c_dim
        return component_pos

    def forecast(self,
                 params,
                 obs_time_series,
                 num_forecast_steps,
                 past_covariates=None,
                 forecast_covariates=None,
                 key=jr.PRNGKey(0)):
        """Forecast the time series.
        """
        if forecast_covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            past_inputs = self.regression(params[self.reg_name], past_covariates)
            forecast_inputs = self.regression(params[self.reg_name], forecast_covariates)
        else:
            past_inputs = jnp.zeros(obs_time_series.shape)
            forecast_inputs = jnp.zeros((num_forecast_steps, self.dim_obs))

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)
        get_trans_mat = partial(self.get_trans_mat, params)
        get_trans_cov = partial(self.get_trans_cov, params)

        # Filtering the observed time series to initialize the forecast
        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        def _step(current_states, args):
            key, forecast_input, t = args
            key1, key2 = jr.split(key)
            cur_mean, cur_cov, cur_state = current_states

            # Observation of the previous time step.
            obs_mean_unc = self.obs_mat @ cur_mean + forecast_input
            obs_mean = self._emission_constrainer(obs_mean_unc)
            obs_cov = self.obs_mat @ cur_cov @ self.obs_mat.T + params['obs_model']['cov']
            obs = self.emission_distribution(cur_state, forecast_input).sample(seed=key2)

            # Predict the latent state of the next time step.
            next_mean = get_trans_mat(t) @ cur_mean
            next_cov = get_trans_mat(t) @ cur_cov @ get_trans_mat(t).T\
                + self.cov_select_mat @ get_trans_cov(t) @ self.cov_select_mat.T
            next_state = get_trans_mat(t) @ cur_state\
                + self.cov_select_mat @ MVN(jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)
            return (next_mean, next_cov, next_state), (obs_mean, obs_cov, obs)

        # The first time step of forecast.
        t0 = obs_time_series.shape[0]
        initial_mean = get_trans_mat(t0) @ filtered_mean[-1]
        initial_cov = get_trans_mat(t0) @ filtered_cov[-1] @ get_trans_mat(t0).T\
            + self.cov_select_mat @ get_trans_cov(t0) @ self.cov_select_mat.T
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)
        initial_states = (initial_mean, initial_cov, initial_state)

        # Forecast the following up steps.
        carrys = (jr.split(key, num_forecast_steps),
                  forecast_inputs,
                  t0 + 1 + jnp.arange(num_forecast_steps))
        _, (ts_means, ts_mean_covs, ts) = lax.scan(_step, initial_states, carrys)

        return ts_means, ts_mean_covs, ts

    def one_step_predict(self, params, obs_time_series, covariates=None):
        """One step forward prediction.
           This is a by product of the Kalman filter.
        """
        raise NotImplementedError

    def get_trans_mat(self, params, t):
        """Evaluate the transition matrix of the latent state at time step t,
           conditioned on parameters of the model.
        """
        trans_mat = []
        for c_name in self.latent_comp_names:
            # Obtain the transition matrix of each single latent component.
            trans_getter = self.trans_mat_getters[c_name]
            c_trans_mat = trans_getter(params[c_name], t)
            trans_mat.append(c_trans_mat)
        return jsp.linalg.block_diag(*trans_mat)

    def get_trans_cov(self, params, t):
        """Evaluate the covariance of the latent dynamics at time step t,
           conditioned on parameters of the model.
        """
        trans_cov = []
        for c_name in self.latent_comp_names:
            # Obtain the covariance of each single latent component.
            cov_getter = self.trans_cov_getters[c_name]
            c_trans_cov = cov_getter(params[c_name], t)
            trans_cov.append(c_trans_cov)
        return jsp.linalg.block_diag(*trans_cov)

    def _to_ssm_params(self, params):
        """Convert the STS model into the form of the corresponding SSM model.
        """
        get_trans_mat = partial(self.get_trans_mat, params)
        get_sparse_cov = lambda t:\
            self.cov_select_mat @ self.get_trans_cov(params, t) @ self.cov_select_mat.T
        if self.obs_distribution == 'Gaussian':
            return ParamsLGSSM(
                initial=ParamsLGSSMInitial(mean=self.initial_mean,
                                           cov=self.initial_cov),
                dynamics=ParamsLGSSMDynamics(weights=get_trans_mat,
                                             bias=jnp.zeros(self.dim_state),
                                             input_weights=jnp.zeros(self.dim_state, 1),
                                             cov=get_sparse_cov),
                emissions=ParamsLGSSMEmissions(weights=self.obs_mat,
                                               bias=jnp.zeros(self.dim_obs),
                                               input_weights=jnp.eye(self.dim_obs),
                                               cov=params['obs_model']['cov'])
                )
        elif self.obs_distribution == 'Poisson':
            # Current formulation of the dynamics function cannot depends on t
            trans_mat = get_trans_mat(0)
            sparse_cov = get_sparse_cov(0)
            return ParamsGGSSM(
                initial_mean=self.initial_mean,
                initial_covariance=self.initial_cov,
                dynamics_function=lambda z: trans_mat @ z,
                dynamics_covariance=sparse_cov,
                emission_mean_function=lambda z: self._emission_constrainer(self.obs_mat @ z),
                emission_cov_function=lambda z: jnp.diag(self._emission_constrainer(self.obs_mat @ z)),
                emission_dist=lambda mu, Sigma: Pois(log_rate=jnp.log(mu))
                )

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model.
        """
        if self.obs_distribution == 'Gaussian':
            return lgssm_filter(params=params,
                                emissions=emissions,
                                inputs=inputs)
        elif self.obs_distribution == 'Poisson':
            return cmgf_filt(model_params=params,
                             inf_params=EKFIntegrals(),
                             emissions=emissions,
                             inputs=inputs,
                             num_iter=2)

    def _ssm_smoother(self, params, emissions, inputs):
        """The smoother of the corresponding SSM model
        """
        if self.obs_distribution == 'Gaussian':
            return lgssm_smoother(params=params, emissions=emissions, inputs=inputs)
        elif self.obs_distribution == 'Poisson':
            return cmgf_smooth(params=params, inf_params=EKFIntegrals(), emissions=emissions,
                               inputs=inputs, num_iter=2)

    def _ssm_posterior_sample(self, ssm_params, obs_time_series, inputs, key):
        """The posterior sampler of the corresponding SSM model
        """
        if self.obs_distribution == 'Gaussian':
            return lgssm_posterior_sample(
                rng=key, params=ssm_params, emissions=obs_time_series, inputs=inputs)
        elif self.obs_distribution == 'Poisson':
            # Currently the posterior_sample for STS model with Poisson likelihood
            # simply returns the filtered means.
            return self._ssm_filter(ssm_params, obs_time_series, inputs)

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space.
        """
        if self.obs_distribution == 'Gaussian':
            return emission
        elif self.obs_distribution == 'Poisson':
            return jnp.exp(emission)
