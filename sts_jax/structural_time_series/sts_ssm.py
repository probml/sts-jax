from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from dynamax.generalized_gaussian_ssm.inference import EKFIntegrals
from dynamax.generalized_gaussian_ssm.inference import (
    iterated_conditional_moments_gaussian_filter as cmgf_filt,
)
from dynamax.generalized_gaussian_ssm.inference import (
    iterated_conditional_moments_gaussian_smoother as cmgf_smooth,
)
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.linear_gaussian_ssm import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_filter,
    lgssm_posterior_sample,
    lgssm_smoother,
)
from dynamax.ssm import SSM
from dynamax.types import PRNGKey, Scalar
from jax import lax, vmap
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)
from tensorflow_probability.substrates.jax.distributions import Poisson as Pois

from .sts_components import ParamPriorsSTS, ParamPropertiesSTS, ParamsSTS

# from dynamax.generalized_gaussian_ssm import (
#    ParamsGGSSM,
#    EKFIntegrals,
#    iterated_conditional_moments_gaussian_filter as cmgf_filt,
#    iterated_conditional_moments_gaussian_smoother as cmgf_smooth
#    )


class StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    """

    def __init__(
        self,
        params: ParamsSTS,
        param_props: ParamPropertiesSTS,
        param_priors: ParamPriorsSTS,
        trans_mat_getters: List,
        trans_cov_getters: List,
        obs_mats: List,
        cov_select_mats: List,
        initial_distributions: List,
        reg_func: Callable = None,
        obs_distribution: str = "Gaussian",
        dim_covariates: int = 0,
    ) -> None:
        """
        Args:
            params: parameters of the STS model, is an instance of OrderedDict, each item is
                parameters of one component.
            param_props: properties of parameters of the STS model, having same tree structure
                with 'params', each leaf node is a instance of 'ParameterProperties' for the
                parameter in the conrresponding leaf node of 'params'.
            param_priors: priors of parameters of the STS model, having same tree structure
                with 'params', each leaf node is a prior distribution for the parameter in the
                corresponding leaf node of 'params'.
            trans_mat_getters: list of transition_matrix_getters, one for each latent component.
            trans_cov_getters: list of nonsingular_transition_covariance_getters, one for each
                latent component.
            obs_mats: list of observation matrices, one for each latent component.
            cov_select_mats: list of transition_covariance_selecting matrices, one for each
                latent component.
            initial_distributions: list of initial distributions for latent state, one for each
                latent component.
            reg_func: regression function of the regression component.
            obs_distribution: distribution family of the observation, can be 'Gaussian' or
                'Poisson'.
            dim_covariates: dimension of the covariates.
        """

        self.params = params
        self.param_props = param_props
        self.param_priors = param_priors

        self.trans_mat_getters = trans_mat_getters
        self.trans_cov_getters = trans_cov_getters
        self.component_obs_mats = obs_mats
        self.cov_select_mats = cov_select_mats

        self.latent_comp_names = cov_select_mats.keys()
        self.obs_distribution = obs_distribution

        # Combine means and covariances of the initial state.
        self.initial_mean = jnp.concatenate([init_pri.mode() for init_pri in initial_distributions.values()])
        self.initial_cov = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in initial_distributions.values()]
        )

        # Combine fixed observation matrices and the covariance selecting matrices.
        self.obs_mat = jnp.concatenate([*obs_mats.values()], axis=1)
        self.cov_select_mat = jsp.linalg.block_diag(*cov_select_mats.values())

        # Dimensions of the observation and the latent state
        self.dim_obs, self.dim_state = self.obs_mat.shape
        # Rank of the latent state
        self.dim_comp = self.get_trans_cov(self.params, 0).shape[0]
        # Dimension of the covariates
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

    def log_prior(self, params: ParamsSTS) -> Scalar:
        """Log prior probability of parameters."""
        lp = 0.0
        for c_name, c_priors in self.param_priors.items():
            for p_name, p_pri in c_priors.items():
                lp += p_pri.log_prob(params[c_name][p_name])
        return lp

    def initial_distribution(self):
        """Distribution of the initial state of the SSM form of the STS model."""
        return MVN(self.initial_mean, self.initial_cov)

    def transition_distribution(self, state):
        """This is a must-have method of SSM.
        Not implemented here because tfp.distribution does not support multivariate normal
        distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(
        self, state: Float[Array, "dim_state"], obs_input: Float[Array, "dim_obs"]
    ) -> tfd.Distribution:
        """Emission distribution of the SSM at one time step.
        The argument 'obs_input' is not the covariate of the STS model, it is either an array
        of 0's or the output of the regression component at the current time step, which will
        be added directly to the observation model.
        """
        if self.obs_distribution == "Gaussian":
            return MVN(self.obs_mat @ state + obs_input, self.params["obs_model"]["cov"])
        elif self.obs_distribution == "Poisson":
            unc_rates = self.obs_mat @ state + obs_input
            return Pois(rate=self._emission_constrainer(unc_rates))

    def sample(
        self,
        params: ParamsSTS,
        num_timesteps: int,
        initial_state: Optional[Float[Array, "dim_states"]] = None,
        initial_timestep: int = 0,
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[Float[Array, "num_timesteps dim_obs"], Float[Array, "num_timesteps dim_obs"]]:
        """Sample a sequence of latent states and emissions with given parameters of the STS.

        Args:
            initial_state: the latent state of the 1st sample.
            initial_timestep: starting time step of sampling, is only used when the transition
                matrix is time-dependent.
        """

        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros((num_timesteps, self.dim_obs))

        key1, key2 = jr.split(key, 2)
        if initial_state is None:
            initial_state = self.initial_distribution.sample(seed=key1)

        get_trans_mat = partial(self.get_trans_mat, params)
        get_trans_cov = partial(self.get_trans_cov, params)

        def _step(curr_state, args):
            key, input, t = args
            key1, key2 = jr.split(key, 2)
            # The latent state of the next time point.
            next_state = get_trans_mat(t) @ curr_state + self.cov_select_mat @ MVN(
                jnp.zeros(self.dim_comp), get_trans_cov(t)
            ).sample(seed=key1)
            curr_obs = self.emission_distribution(curr_state, input).sample(seed=key2)
            return next_state, (curr_state, curr_obs)

        # Sample the following emissions and states.
        keys = jr.split(key2, num_timesteps)
        _, (states, sample_obs) = lax.scan(
            _step, initial_state, (keys, inputs, initial_timestep + jnp.arange(num_timesteps))
        )
        sample_mean = self._emission_constrainer(states @ self.obs_mat.T + inputs)
        return sample_mean, sample_obs

    def marginal_log_prob(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
    ) -> Scalar:
        """Compute marginal log likelihood of the observed time series given model parameters."""
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the model to SSM and perform filtering.
        ssm_params = self._to_ssm_params(params)
        filtered_posterior = self._ssm_filter(params=ssm_params, emissions=obs_time_series, inputs=inputs)
        return filtered_posterior.marginal_loglik

    def posterior_sample(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[Float[Array, "num_timesteps dim_obs"], Float[Array, "num_timesteps dim_obs"]]:
        """Sample latent states from the posterior distribution, as well as the predictive
        observations, given model parameters.
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

        # Sample predictive observations conditioned on posterior samples of latent states.
        obs_sampler = lambda state, input, key: self.emission_distribution(state, input).sample(seed=key)
        keys = jr.split(key2, obs_time_series.shape[0])

        predictive_obs = vmap(obs_sampler)(states, inputs, keys)
        predictive_mean = self._emission_constrainer(states @ self.obs_mat.T + inputs)
        return predictive_mean, predictive_obs

    def component_posterior(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
    ) -> OrderedDict:
        """Decompose the STS model into components and return the means and variances
        of the marginal posterior of each component.
        """
        component_pos = OrderedDict()
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
            # Add the component corresponding to the regression component, which has no variance.
            component_pos[self.reg_name] = {
                "pos_mean": inputs,
                "pos_cov": jnp.zeros((*obs_time_series.shape, self.dim_obs)),
            }
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
            # Extract posterior mean and covariances of each component from the latent state.
            c_dim = obs_mat.shape[1]
            c_mean = mu_pos[:, _loc : _loc + c_dim]
            c_cov = var_pos[:, _loc : _loc + c_dim, _loc : _loc + c_dim]
            # Posterior emission of the single component.
            c_obs_mean_unc = vmap(jnp.matmul, (None, 0))(obs_mat, c_mean)
            c_obs_mean = self._emission_constrainer(c_obs_mean_unc)
            if self.obs_distribution == "Gaussian":
                c_obs_cov = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_cov, obs_mat)
            elif self.obs_distribution == "Poisson":
                # Set the covariance to be 0 if the distribution of the observation is Poisson.
                c_obs_cov = jnp.zeros((*obs_time_series.shape, self.dim_obs))
            component_pos[c] = {"pos_mean": c_obs_mean, "pos_cov": c_obs_cov}
            _loc += c_dim
        return component_pos

    def forecast(
        self,
        params: ParamsSTS,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
        num_forecast_steps: int,
        num_forecast_samples: int = 100,
        past_covariates: Optional[Float[Array, "num_timesteps dim_covariates"]] = None,
        forecast_covariates: Optional[Float[Array, "num_forecast_steps dim_covariates"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[
        Float[Array, "num_forecast_samples num_forecast_steps dim_obs"],
        Float[Array, "num_forecast_samples num_forecast_steps dim_obs"],
    ]:
        """Forecast the time series."""
        if forecast_covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            past_inputs = self.regression(params[self.reg_name], past_covariates)
        else:
            past_inputs = jnp.zeros(obs_time_series.shape)

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)
        get_trans_mat = partial(self.get_trans_mat, params)
        get_trans_cov = partial(self.get_trans_cov, params)

        # Filtering the observed time series to initialize the forecast
        filtered_posterior = self._ssm_filter(params=ssm_params, emissions=obs_time_series, inputs=past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        # The first time step of forecast.
        t0 = obs_time_series.shape[0] - 1
        initial_state_mean = get_trans_mat(t0) @ filtered_mean[-1]
        initial_state_cov = (
            get_trans_mat(t0) @ filtered_cov[-1] @ get_trans_mat(t0).T
            + self.cov_select_mat @ get_trans_cov(t0) @ self.cov_select_mat.T
        )
        initial_states = MVN(initial_state_mean, initial_state_cov).sample(num_forecast_samples, seed=key)

        # Forecast by sample from an STS model conditioned on the parameter and initialized
        # using the filtered posterior.
        single_forecast = lambda initial_state, key: self.sample(
            params, num_forecast_steps, initial_state, t0, forecast_covariates, key
        )

        forecast_mean, forecast_obs = vmap(single_forecast)(initial_states, jr.split(key, num_forecast_samples))
        return forecast_mean, forecast_obs

    def one_step_predict(self, params, obs_time_series, covariates=None):
        """One step forward prediction.
        This is a by product of the Kalman filter.
        A general method of one-step-forward prediction is to be added to the class
        dynamax.LinearGaussianSSM
        """
        raise NotImplementedError

    def get_trans_mat(self, params: ParamsSTS, t: int) -> Float[Array, "dim_state dim_state"]:
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

    def get_trans_cov(self, params: ParamsSTS, t: int) -> Float[Array, "order_state order_state"]:
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
        """Convert the STS model into the form of the corresponding SSM model."""
        get_trans_mat = partial(self.get_trans_mat, params)
        get_sparse_cov = lambda t: self.cov_select_mat @ self.get_trans_cov(params, t) @ self.cov_select_mat.T
        if self.obs_distribution == "Gaussian":
            return ParamsLGSSM(
                initial=ParamsLGSSMInitial(mean=self.initial_mean, cov=self.initial_cov),
                dynamics=ParamsLGSSMDynamics(
                    weights=get_trans_mat,
                    bias=jnp.zeros(self.dim_state),
                    input_weights=jnp.zeros((self.dim_state, 1)),
                    cov=get_sparse_cov,
                ),
                emissions=ParamsLGSSMEmissions(
                    weights=self.obs_mat,
                    bias=jnp.zeros(self.dim_obs),
                    input_weights=jnp.eye(self.dim_obs),
                    cov=params["obs_model"]["cov"],
                ),
            )
        elif self.obs_distribution == "Poisson":
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
                emission_dist=lambda mu, Sigma: Pois(rate=mu),
            )

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model."""
        if self.obs_distribution == "Gaussian":
            return lgssm_filter(params=params, emissions=emissions, inputs=inputs)
        elif self.obs_distribution == "Poisson":
            return cmgf_filt(
                model_params=params, inf_params=EKFIntegrals(), emissions=emissions, inputs=inputs, num_iter=2
            )

    def _ssm_smoother(self, params, emissions, inputs):
        """The smoother of the corresponding SSM model"""
        if self.obs_distribution == "Gaussian":
            return lgssm_smoother(params=params, emissions=emissions, inputs=inputs)
        elif self.obs_distribution == "Poisson":
            return cmgf_smooth(params=params, inf_params=EKFIntegrals(), emissions=emissions, inputs=inputs, num_iter=2)

    def _ssm_posterior_sample(self, ssm_params, obs_time_series, inputs, key):
        """The posterior sampler of the corresponding SSM model"""
        if self.obs_distribution == "Gaussian":
            return lgssm_posterior_sample(key=key, params=ssm_params, emissions=obs_time_series, inputs=inputs)
        elif self.obs_distribution == "Poisson":
            # Currently the posterior_sample for STS model with Poisson likelihood
            # simply returns the filtered means.
            return self._ssm_filter(ssm_params, obs_time_series, inputs).filtered_means

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space."""
        if self.obs_distribution == "Gaussian":
            return emission
        elif self.obs_distribution == "Poisson":
            return jnp.exp(emission)
