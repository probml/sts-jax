import jax.numpy as jnp
import jax.random as jr
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from jax import lax
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

from sts_jax.structural_time_series.sts_components import Autoregressive
from sts_jax.structural_time_series.sts_model import StructuralTimeSeries as STS


def _build_models(time_steps, key):

    keys = jr.split(key, 5)
    standard_mvn = MVN(jnp.zeros(1), jnp.eye(1))

    # Generate parameters of the STS component
    level_scale = 5.0
    coef = 0.8
    initial_level = standard_mvn.sample(seed=keys[0])

    obs_noise_scale = 4.0

    # Generate observed time series using the SSM representation.
    F = jnp.array([[coef]])
    H = jnp.array([[1]])
    Q = jnp.array([[level_scale]])
    R = obs_noise_scale

    def _step(current_state, key):
        key1, key2 = jr.split(key)
        current_obs = H @ current_state + R * standard_mvn.sample(seed=key1)
        next_state = F @ current_state + Q @ MVN(jnp.zeros(1), jnp.eye(1)).sample(seed=key2)
        return next_state, current_obs

    initial_state = initial_level
    key_seq = jr.split(keys[2], time_steps)
    _, obs_time_series = lax.scan(_step, initial_state, key_seq)

    # Build the STS model using tfp module.
    tfp_comp = tfp.sts.Autoregressive(order=1, observed_time_series=obs_time_series, name="ar")
    tfp_model = tfp.sts.Sum([tfp_comp], observed_time_series=obs_time_series)

    # Build the dynamax STS model.
    dynamax_comp = Autoregressive(order=1, name="ar")
    dynamax_model = STS([dynamax_comp], obs_time_series=obs_time_series)

    # Set the parameters to the parameters learned by the tfp module and fix the parameters.
    tfp_vi_posterior = tfp.sts.build_factored_surrogate_posterior(tfp_model)
    tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=tfp_model.joint_distribution(obs_time_series).log_prob,
        surrogate_posterior=tfp_vi_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=200,
        jit_compile=True,
    )
    vi_dists, _ = tfp_vi_posterior.distribution.sample_distributions()
    tfp_params = tfp_vi_posterior.sample(sample_shape=(1,))

    dynamax_model.params["ar"]["cov_level"] = jnp.atleast_2d(jnp.array(tfp_params["ar/_level_scale"] ** 2))
    dynamax_model.params["ar"]["coef"] = jnp.array(tfp_params["ar/_coefficients"])[0]
    dynamax_model.params["obs_model"]["cov"] = jnp.atleast_2d(jnp.array(tfp_params["observation_noise_scale"] ** 2))

    return (tfp_model, tfp_params, dynamax_model, dynamax_model.params, obs_time_series, vi_dists)


def test_autoregress(time_steps=150, key=jr.PRNGKey(3)):

    tfp_model, tfp_params, dynamax_model, dynamax_params, obs_time_series, vi_dists = _build_models(time_steps, key)

    # Fit and forecast with the tfp module.
    # Not use tfp.sts.decompose_by_component() since its output series is centered at 0.
    masked_time_series = tfp.sts.MaskedTimeSeries(
        time_series=obs_time_series, is_missing=tf.math.is_nan(obs_time_series)
    )
    tfp_posterior = tfp.sts.impute_missing_values(
        tfp_model, masked_time_series, tfp_params, include_observation_noise=False
    )

    tfp_posterior_mean = jnp.array(tfp_posterior.mean()).squeeze()
    tfp_posterior_scale = jnp.array(jnp.array(tfp_posterior.stddev())).squeeze()

    # Fit and forecast with dynamax
    dynamax_posterior = dynamax_model.decompose_by_component(dynamax_params, obs_time_series)
    dynamax_posterior_mean = dynamax_model._uncenter_obs(dynamax_posterior["ar"]["pos_mean"]).squeeze()
    dynamax_posterior_cov = dynamax_posterior["ar"]["pos_cov"].squeeze()

    # Compare posterior inference by tfp and dynamax.
    # In comparing the smoothed posterior, we omit the first 5 time steps,
    # since the tfp and the dynamax implementations of STS has different settings in
    # distributions of initial state, which will influence the posterior inference of
    # the first few states.
    len_step = jnp.abs(tfp_posterior_mean[1:] - tfp_posterior_mean[:-1]).mean()
    assert jnp.allclose(tfp_posterior_mean[5:], dynamax_posterior_mean[5:], atol=len_step)
    assert jnp.allclose(tfp_posterior_scale[5:], jnp.sqrt(dynamax_posterior_cov)[5:], rtol=1e-2)


@pytest.mark.skip(
    reason="Skipped because the forecast mean and variances are now computed as sample mean and variance, for dynamax model"
)
def test_autoregress_forecast(time_steps=150, key=jr.PRNGKey(3)):
    tfp_model, tfp_params, dynamax_model, dynamax_params, obs_time_series, vi_dists = _build_models(time_steps, key)

    masked_time_series = tfp.sts.MaskedTimeSeries(
        time_series=obs_time_series, is_missing=tf.math.is_nan(obs_time_series)
    )

    tfp_posterior = tfp.sts.impute_missing_values(
        tfp_model, masked_time_series, tfp_params, include_observation_noise=False
    )

    tfp_posterior_mean = jnp.array(tfp_posterior.mean()).squeeze()

    tfp_forecasts = tfp.sts.forecast(
        tfp_model, obs_time_series, parameter_samples=tfp_params, num_steps_forecast=50, include_observation_noise=True
    )

    tfp_forecast_mean = jnp.array(tfp_forecasts.mean()).squeeze()
    tfp_forecast_scale = jnp.array(tfp_forecasts.stddev()).squeeze()

    dynamax_forecast = dynamax_model.forecast(dynamax_params, obs_time_series, num_forecast_steps=50)[1]
    dynamax_forecast_mean = jnp.concatenate(dynamax_forecast).mean(axis=0).squeeze()
    dynamax_forecast_cov = jnp.concatenate(dynamax_forecast).var(axis=0).squeeze()
    # Compare forecast by tfp and dynamax.
    len_step = jnp.abs(tfp_posterior_mean[1:] - tfp_posterior_mean[:-1]).mean()
    assert jnp.allclose(tfp_forecast_mean, dynamax_forecast_mean, atol=0.5 * len_step)
    assert jnp.allclose(tfp_forecast_scale, jnp.sqrt(dynamax_forecast_cov), rtol=5e-2)
