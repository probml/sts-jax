import jax.numpy as jnp
import jax.random as jr
from jax import lax

from dynamax.structural_time_series.models.sts_model import StructuralTimeSeries as STS
from dynamax.structural_time_series.models.sts_components import LocalLinearTrend

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN
)


def _build_models(time_steps, key):

    keys = jr.split(key, 5)
    standard_mvn = MVN(jnp.zeros(1), jnp.eye(1))

    # Generate covariates from an AR(1) model
    c0 = standard_mvn.sample(seed=keys[0])
    cf = jnp.array([[0.8]])
    cq = jnp.array([[5.]])

    def _step(c0, key):
        c1 = cf @ c0 + cq @ standard_mvn.sample(seed=key)
        return c1, c0
    _keys = jr.split(keys[1], time_steps + 50)
    _, covariates = lax.scan(_step, c0, _keys)

    weights = jnp.array([[5.], [0]]) + standard_mvn.sample(seed=keys[2], sample_shape=(2,))

    # Generate parameters of the STS component
    inputs = jnp.concatenate((covariates, jnp.ones((time_steps + 50, 1))), axis=1)
    obs_noise_scale = 6.

    # Generate observed time series.
    obs_mean = inputs[:time_steps] @ weights
    obs_time_series = obs_mean + obs_noise_scale*standard_mvn.sample(seed=keys[3],
                                                                     sample_shape=(time_steps,))

    # Build the STS model using tfp module.
    tfp_comp = tfp.sts.LinearRegression(inputs, name='linear_regression')
    tfp_model = tfp.sts.Sum([tfp_comp], observed_time_series=obs_time_series)

    # Build the dynamax STS model.
    dynamax_comp = LinearRegression(dim_covariates=1, name='linear_regression')
    dynamax_model = STS([dynamax_comp],
                        obs_time_series=obs_time_series, covariates=covariates[:time_steps])

    # Set the parameters to the parameters learned by the tfp module and fix the parameters.
    tfp_vi_posterior = tfp.sts.build_factored_surrogate_posterior(tfp_model)
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=tfp_model.joint_distribution(obs_time_series).log_prob,
        surrogate_posterior=tfp_vi_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=200,
        jit_compile=True)
    vi_dists, _ = tfp_vi_posterior.distribution.sample_distributions()
    tfp_params = tfp_vi_posterior.sample(sample_shape=(1,))

    dynamax_model.params['linear_regression']['weights'] =\
        jnp.atleast_2d(jnp.array(tfp_params['linear_regression/_weights'])).T
    dynamax_model.params['obs_model']['cov'] =\
        jnp.atleast_2d(jnp.array(tfp_params['observation_noise_scale']**2))

    return (tfp_model, tfp_params,
            dynamax_model, dynamax_model.params,
            covariates,
            obs_time_series,
            vi_dists)


def test_local_linear_trend_forecast(time_steps=150, key=jr.PRNGKey(3)):

    tfp_model, tfp_params, dynamax_model, dynamax_params, covariates, obs_time_series, vi_dists =\
        _build_models(time_steps, key)
    covariates_pre = covariates[:time_steps]
    covariates_pos = covariates[time_steps:]

    # Fit and forecast with the tfp module.
    # Not use tfp.sts.decmopose_by_component() since its output series is centered at 0.
    masked_time_series = tfp.sts.MaskedTimeSeries(time_series=obs_time_series,
                                                  is_missing=tf.math.is_nan(obs_time_series))
    tfp_posterior = tfp.sts.impute_missing_values(tfp_model, masked_time_series, tfp_params,
                                                  include_observation_noise=False)
    tfp_forecasts = tfp.sts.forecast(tfp_model, obs_time_series,
                                     parameter_samples=tfp_params,
                                     num_steps_forecast=50,
                                     include_observation_noise=True)
    tfp_posterior_mean = jnp.array(tfp_posterior.mean()).squeeze()
    tfp_posterior_scale = jnp.array(jnp.array(tfp_posterior.stddev())).squeeze()
    tfp_forecast_mean = jnp.array(tfp_forecasts.mean()).squeeze()
    tfp_forecast_scale = jnp.array(tfp_forecasts.stddev()).squeeze()

    # Fit and forecast with dynamax
    dynamax_posterior = dynamax_model.decompose_by_component(dynamax_params,
                                                             obs_time_series,
                                                             covariates=covariates_pre)
    dynamax_forecast = dynamax_model.forecast(dynamax_params, obs_time_series,
                                              num_forecast_steps=50,
                                              past_covariates=covariates_pre,
                                              forecast_covariates=covariates_pos)
    dynamax_posterior_mean = dynamax_posterior['local_linear_trend']['pos_mean'].squeeze()
    dynamax_posterior_cov = dynamax_posterior['local_linear_trend']['pos_cov'].squeeze()
    dynamax_forecast_mean = dynamax_forecast['means'].squeeze()
    dynamax_forecast_cov = dynamax_forecast['covariances'].squeeze()

    # Compare posterior inference by tfp and dynamax.
    # In comparing the smoothed posterior, we omit the first 5 time steps,
    # since the tfp and the dynamax implementations of STS has different settings in
    # distributions of initial state, which will influence the posterior inference of
    # the first few states.
    len_step = jnp.abs(tfp_posterior_mean[1:]-tfp_posterior_mean[:-1]).mean()
    assert jnp.allclose(tfp_posterior_mean[5:], dynamax_posterior_mean[5:], atol=len_step)
    assert jnp.allclose(tfp_posterior_scale[5:], jnp.sqrt(dynamax_posterior_cov)[5:], rtol=1e-1)
    # Compoare forecast by tfp and dynamax.
    assert jnp.allclose(tfp_forecast_mean, dynamax_forecast_mean, atol=0.5*len_step)
    assert jnp.allclose(tfp_forecast_scale, jnp.sqrt(dynamax_forecast_cov), rtol=5e-2)
