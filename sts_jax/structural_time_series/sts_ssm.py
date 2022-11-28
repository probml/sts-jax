from jax import lax, vmap
from collections import OrderedDict
from functools import partial
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Float, Array
from dynamax.ssm import SSM
from dynamax.types import PRNGKey, Scalar
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
    # lgssm_filter,
    # lgssm_smoother,
    # lgssm_posterior_sample
    )
from .inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from .sts_components import ParamsSTS, ParamPropertiesSTS, ParamPriorsSTS
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    Poisson as Pois)
from typing import List, Callable, Optional


class StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    """

    def __init__(self,
                 params: ParamsSTS,
                 param_props: ParamPropertiesSTS,
                 param_priors: ParamPriorsSTS,
                 trans_mat_getters: List,
                 trans_cov_getters: List,
                 obs_mats: List,
                 cov_select_mats: List,
                 initial_distributions: List,
                 reg_func: Callable=None,
                 obs_distribution: str='Gaussian',
                 dim_covariates: int=0) -> None:
        self.params = params
        self.param_props = param_props
        self.param_priors = param_priors

        self.trans_mat_getters = trans_mat_getters
        self.trans_cov_getters = trans_cov_getters
        self.component_obs_mats = obs_mats
        self.cov_select_mats = cov_select_mats

        self.latent_comp_names = cov_select_mats.keys()
        self.obs_distribution = obs_distribution

        # Mean and covariance of the initial state.
        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in initial_distributions.values()])
        self.initial_cov = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in initial_distributions.values()])

        # Fixed observation matrix and the covariance selection matrix.
        self.obs_mat = jnp.concatenate([*obs_mats.values()], axis=1)
        self.cov_select_mat = jsp.linalg.block_diag(*cov_select_mats.values())

        # Dimensions of the SSM.
        self.dim_obs, self.dim_state = self.obs_mat.shape
        self.dim_comp = self.get_trans_cov(self.params, 0).shape[0]
        self.dim_covariates = dim_covariates

        # Pick out the regression component if there is one.
        if reg_func is not None:
            # Regression component is always the last component if there is one.
            self.reg_name = list(params.keys())[-1]
            self.regression = reg_func

    @property
    def emission_shape(self):
        return (self.dim_obs,)

    @property
    def inputs_shape(self):
        return (self.dim_covariates,)

    def log_prior(
        self,
        params: ParamsSTS
    ) -> Scalar:
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
        """Not implemented because tfp.distribution does not support
           multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(
        self,
        state,
        inputs
    ) -> tfd.Distribution:
        """Emission distribution of the SSM at one time step.
        """
        if self.obs_distribution == 'Gaussian':
            return MVN(self.obs_mat @ state + inputs, self.params['obs_model']['cov'])
        elif self.obs_distribution == 'Poisson':
            unc_rates = self.obs_mat @ state + inputs
            return Pois(rate=self._emission_constrainer(unc_rates))

    def sample(
        self,
        params: ParamsSTS,
        num_timesteps: int,
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Float[Array, "num_timesteps dim_obs"]:
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

        def _step(curr_state, args):
            key, input, t = args
            key1, key2 = jr.split(key, 2)
            # The latent state of the next time point.
            next_state = get_trans_mat(t) @ curr_state + self.cov_select_mat @ MVN(
                jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)
            curr_obs = self.emission_distribution(curr_state, input).sample(seed=key2)
            return next_state, curr_obs

        # Sample the initial state.
        key1, key2 = jr.split(key, 2)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the following emissions and states.
        keys = jr.split(key2, num_timesteps)
        _, time_series = lax.scan(
            _step, initial_state, (keys, inputs, jnp.arange(num_timesteps)))
        return time_series

    def marginal_log_prob(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None
    ) -> Scalar:
        """Compute marginal log likelihood of the observed time series given model parameters.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the model to SSM and perform filtering.
        ssm_params = self._to_ssm_params(params)
        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=inputs)
        return filtered_posterior.marginal_loglik

    def posterior_sample(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Float[Array, "num_timesteps dim_obs"]:
        """Sample latent states from the posterior given model parameters.
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
        states = self._ssm_posterior_sample(ssm_params, obs_time_series, inputs, key1)

        # Sample observations.
        keys = jr.split(key2, obs_time_series.shape[0])
        obs_sampler = lambda state, input, key:\
            self.emission_distribution(state, input).sample(seed=key)
        time_series = vmap(obs_sampler)(states, inputs, keys)
        return time_series

    def component_posterior(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None
    ) -> OrderedDict:
        """Decompose the STS model into components and return the means and variances
           of the marginal posterior of each component.
        """
        component_pos = OrderedDict()
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
            component_pos[self.reg_name] = {
                'pos_mean': inputs,
                'pos_cov': jnp.zeros((*obs_time_series.shape, self.dim_obs))}
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
            c_obs_mean_unc = vmap(jnp.matmul, (None, 0))(obs_mat, c_mean)
            c_obs_mean = self._emission_constrainer(c_obs_mean_unc)
            if self.obs_distribution == 'Gaussian':
                c_obs_cov = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_cov, obs_mat)
            elif self.obs_distribution == 'Poisson':
                # Set the covariance to be 0 if the distribution of the observation is Poisson.
                c_obs_cov = jnp.zeros((*obs_time_series.shape, self.dim_obs))
            component_pos[c] = {'pos_mean': c_obs_mean, 'pos_cov': c_obs_cov}
            _loc += c_dim
        return component_pos

    # def forecast(self,
    #              params: ParamsSTS,
    #              obs_time_series: Float[Array, "num_timesteps dim_obs"],
    #              num_forecast_steps: int,
    #              past_covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
    #              forecast_covariates: Optional[Float[Array, "num_forecast_steps dim_covariates"]]=None,
    #              key: PRNGKey=jr.PRNGKey(0)) -> Float[
    #                  Array, "num_forecast_samples num_forecast_steps dim_obs"]:
    #     """Forecast the time series.
    #     """
    #     if forecast_covariates is not None:
    #         # If there is a regression component, set the inputs of the emission model
    #         # of the SSM as the fitted value of the regression component.
    #         past_inputs = self.regression(params[self.reg_name], past_covariates)
    #         forecast_inputs = self.regression(params[self.reg_name], forecast_covariates)
    #     else:
    #         past_inputs = jnp.zeros(obs_time_series.shape)
    #         forecast_inputs = jnp.zeros((num_forecast_steps, self.dim_obs))

    #     # Convert the STS model to SSM.
    #     ssm_params = self._to_ssm_params(params)
    #     get_trans_mat = partial(self.get_trans_mat, params)
    #     get_trans_cov = partial(self.get_trans_cov, params)

    #     # Filtering the observed time series to initialize the forecast
    #     filtered_posterior = self._ssm_filter(
    #         params=ssm_params, emissions=obs_time_series, inputs=past_inputs)
    #     filtered_mean = filtered_posterior.filtered_means
    #     filtered_cov = filtered_posterior.filtered_covariances

    #     if self.obs_distribution == 'Gaussian':
    #         get_obs_cov = lambda curr_state_cov: self.obs_mat @ curr_state_cov @ self.obs_mat.T\
    #             + params['obs_model']['cov']
    #     elif self.obs_distribution == 'Poisson':
    #         # Set covariance to be 0 if the distribution is Poisson.
    #         get_obs_cov = lambda curr_state_cov: jnp.zeros((self.dim_obs, self.dim_obs))

    #     def _step(current_states, args):
    #         key, forecast_input, t = args
    #         key1, key2 = jr.split(key)
    #         curr_state_mean, curr_state_cov, curr_state = current_states

    #         # Observation of the current time step.
    #         obs_mean_unc = self.obs_mat @ curr_state_mean + forecast_input
    #         obs_mean = self._emission_constrainer(obs_mean_unc)
    #         obs_cov = get_obs_cov(curr_state_cov)
    #         obs = self.emission_distribution(curr_state, forecast_input).sample(seed=key2)

    #         # Predict the latent state of the next time step.
    #         next_state_mean = get_trans_mat(t) @ curr_state_mean
    #         next_state_cov = get_trans_mat(t) @ curr_state_cov @ get_trans_mat(t).T\
    #             + self.cov_select_mat @ get_trans_cov(t) @ self.cov_select_mat.T
    #         next_state = get_trans_mat(t) @ curr_state\
    #             + self.cov_select_mat @ MVN(jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)

    #         return (next_state_mean, next_state_cov, next_state), (obs_mean, obs_cov, obs)

    #     # The first time step of forecast.
    #     t0 = obs_time_series.shape[0]
    #     initial_state_mean = get_trans_mat(t0) @ filtered_mean[-1]
    #     initial_state_cov = get_trans_mat(t0) @ filtered_cov[-1] @ get_trans_mat(t0).T\
    #         + self.cov_select_mat @ get_trans_cov(t0) @ self.cov_select_mat.T
    #     initial_state = MVN(initial_state_mean, initial_state_cov).sample(seed=key)
    #     _states = (initial_state_mean, initial_state_cov, initial_state)

    #     # Forecast the following up steps.
    #     carrys = (jr.split(key, num_forecast_steps),
    #               forecast_inputs,
    #               t0 + 1 + jnp.arange(num_forecast_steps))
    #     _, (ts_means, ts_mean_covs, ts) = lax.scan(_step, _states, carrys)

    #     return ts_means, ts_mean_covs, ts

    def forecast(self,
                 params: ParamsSTS,
                 obs_time_series: Float[Array, "num_timesteps dim_obs"],
                 num_forecast_steps: int,
                 num_forecast_samples: int=100,
                 past_covariates: Optional[Float[Array, "num_timesteps dim_covariates"]]=None,
                 forecast_covariates: Optional[Float[Array, "num_forecast_steps dim_covariates"]]=None,
                 key: PRNGKey=jr.PRNGKey(0)) -> Float[
                     Array, "num_forecast_samples num_forecast_steps dim_obs"]:
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

        def _step(curr_state, args):
            key, input, t = args
            key1, key2 = jr.split(key)
            # Observation of the current time step.
            curr_obs = self.emission_distribution(curr_state, input).sample(seed=key2)
            # Predict the latent state of the next time step.
            next_state = get_trans_mat(t) @ curr_state + self.cov_select_mat @ MVN(
                jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)
            return next_state, curr_obs

        # The first time step of forecast.
        t0 = obs_time_series.shape[0]
        initial_state_mean = get_trans_mat(t0) @ filtered_mean[-1]
        initial_state_cov = get_trans_mat(t0) @ filtered_cov[-1] @ get_trans_mat(t0).T\
            + self.cov_select_mat @ get_trans_cov(t0) @ self.cov_select_mat.T
        initial_states = MVN(initial_state_mean,
                             initial_state_cov).sample(num_forecast_samples, seed=key)

        def single_forecast(initial_state, key):
            args = (jr.split(key, num_forecast_steps), forecast_inputs,
                    t0 + 1 + jnp.arange(num_forecast_steps))
            _, forecast = lax.scan(_step, initial_state, args)
            return forecast

        return vmap(single_forecast)(initial_states, jr.split(key, num_forecast_samples))

    def one_step_predict(self, params, obs_time_series, covariates=None):
        """One step forward prediction.
           This is a by product of the Kalman filter.
           A general method of one-step-forward prediction is to be added to the class
           dynamax.LinearGaussianSSM
        """
        raise NotImplementedError

    def get_trans_mat(
        self,
        params: ParamsSTS,
        t: int
    ) -> Float[Array, "dim_state dim_state"]:
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

    def get_trans_cov(
        self,
        params: ParamsSTS,
        t: int
    ) -> Float[Array, "order_state order_state"]:
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
                                             input_weights=jnp.zeros((self.dim_state, 1)),
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
                dynamics_function=lambda z, u: trans_mat @ z,
                dynamics_covariance=sparse_cov,
                emission_mean_function=lambda z, u: self._emission_constrainer(self.obs_mat @ z + u),
                emission_cov_function=lambda z, u: jnp.diag(self._emission_constrainer(self.obs_mat @ z)),
                emission_dist=lambda mu, Sigma: Pois(rate=mu)
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
                key=key, params=ssm_params, emissions=obs_time_series, inputs=inputs)
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
