import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
import matplotlib.pyplot as plt
from typing import Optional
from dynamax.types import PRNGKey

import sts_jax.structural_time_series.sts_model as sts


class CausalImpact():
    """A wrapper of help functions of the causal impact
    """
    def __init__(self, sts_model, intervention_time, predict, causal_impact, observed_timeseries):
        """
        Args:
            sts_model    : an object of the StructualTimeSeries class
            causal_impact: a dict returned by the function 'causal impact'
        """
        self.intervention_time = intervention_time
        self.sts_model = sts_model
        self.predict_point = predict['pointwise']
        self.predict_interval = predict['interval']
        self.impact_point = causal_impact['pointwise']
        self.impact_cumulat = causal_impact['cumulative']
        self.timeseries = observed_timeseries

    def plot(self):
        """Plot the causal impact
        """
        x = jnp.arange(self.timeseries.shape[0])
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(9, 6), sharex=True, layout='constrained')

        # Plot the original obvervation and the counterfactual predict
        ax1.plot(x, self.timeseries, color='black', lw=2, label='Observation')
        ax1.plot(x, self.predict_point, linestyle='dashed', color='blue', lw=2, label='Prediction')
        ax1.fill_between(x, self.predict_interval[0], self.predict_interval[1],
                         color='blue', alpha=0.2)
        ax1.axvline(x=self.intervention_time, linestyle='dashed', color='gray', lw=2)
        ax1.set_title('Original time series')

        # Plot the pointwise causal impact
        ax2.plot(x, self.impact_point[0], linestyle='dashed', color='blue')
        ax2.fill_between(x, self.impact_point[1][0], self.impact_point[1][1],
                         color='blue', alpha=0.2)
        ax2.axvline(x=self.intervention_time, linestyle='dashed', color='gray', lw=2)
        ax2.set_title('Poinwise causal impact')

        # Plot the cumulative causal impact
        ax3.plot(x, self.impact_cumulat[0], linestyle='dashed', color='blue')
        ax3.fill_between(x, self.impact_cumulat[1][0], self.impact_cumulat[1][1],
                         color='blue', alpha=0.2)
        ax3.axvline(x=self.intervention_time, linestyle='dashed', color='gray', lw=2)
        ax3.set_title('Cumulative causal impact')

        plt.show()

    def summary(self):
        msg = "The posterior causl effect"
        print(msg)


def causal_impact(
    obs_time_series: Float[Array, "num_timesteps dim_obs"],
    intervention_timepoint: int,
    obs_distribution: str='Gaussian',
    covariates: Optional[Float[Array, "num_timesteps dim_obs"]]=None,
    sts_model: sts.StructuralTimeSeries=None,
    confidence_level: Float=0.95,
    key: PRNGKey=jr.PRNGKey(0),
    num_samples: int=200
) -> CausalImpact:
    """Inferring the causal impact of an intervention on a time series,
    given the observed timeseries before and after the intervention.

    The causal effect is obtained by conditioned on, and only on, the observations,
    with parameters and latent states integrated out.

    Returns:
        An object of the CausalImpact class
    """
    assert obs_distribution in ['Gaussian', 'Poisson']
    if sts_model is not None:
        assert obs_distribution == sts_model.obs_distribution

    key1, key2, key3 = jr.split(key, 3)
    num_timesteps, dim_obs = obs_time_series.shape

    # Split the data into pre-intervention period and post-intervention period
    time_series_pre = obs_time_series[:intervention_timepoint]
    time_series_pos = obs_time_series[intervention_timepoint:]

    if covariates is not None:
        dim_covariates = covariates.shape[-1]
        # The number of time steps of input must equal to that of observed time_series.
        covariates_pre = covariates[:intervention_timepoint]
        covariates_pos = covariates[intervention_timepoint:]

    # Construct a STS model with only local linear trend by default
    if sts_model is None:
        local_linear_trend = sts.LocalLinearTrend()
        if covariates is None:
            sts_model = sts.StructuralTimeSeries(components=[local_linear_trend],
                                                 obs_time_series=obs_time_series,
                                                 obs_distribution=obs_distribution)
        else:
            linear_regression = sts.LinearRegression(dim_covariates=dim_covariates)
            sts_model = sts.StructuralTimeSeries(components=[local_linear_trend, linear_regression],
                                                 obs_time_series=obs_time_series,
                                                 obs_distribution=obs_distribution,
                                                 covariates=covariates)

    # Fit the STS model, sample from the past and forecast.
    if covariates is not None:
        # Model fitting
        print('Fit the model using HMC...')
        params_posterior_samples, _ = sts_model.fit_hmc(num_samples, time_series_pre,
                                                        covariates=covariates_pre, key=key1)
        print("Model fitting completed.")
        # Sample from the past and forecast
        samples_pre = sts_model.posterior_sample(
            params_posterior_samples, time_series_pre, covariates_pre, key=key2)
        samples_pos = sts_model.forecast(
            params_posterior_samples, time_series_pre, time_series_pos.shape[0],
            covariates_pre, covariates_pos, key3)
    else:
        # Model fitting
        print('Fit the model using HMC...')
        params_posterior_samples, _ = sts_model.fit_hmc(num_samples, time_series_pre, key=key1)
        print("Model fitting completed.")
        # Sample from the past and forecast
        samples_pre = sts_model.posterior_sample(params_posterior_samples, time_series_pre, key=key2)
        samples_pos = sts_model.forecast(
            params_posterior_samples, time_series_pre, time_series_pos.shape[0], key=key3)

    # forecast_means = jnp.concatenate((samples_pre['means'], samples_pos['means']), axis=1).squeeze()
    forecast_observations = jnp.concatenate(
        (samples_pre['observations'], samples_pos['observations']), axis=1).squeeze()

    confidence_bounds = jnp.quantile(
        forecast_observations,
        jnp.array([0.5 - confidence_level/2., 0.5 + confidence_level/2.]),
        axis=0)
    predict_point = forecast_observations.mean(axis=0)
    predict_interval_upper = confidence_bounds[0]
    predict_interval_lower = confidence_bounds[1]

    cum_predict_point = jnp.cumsum(predict_point)
    cum_confidence_bounds = jnp.quantile(
        forecast_observations.cumsum(axis=1),
        jnp.array([0.5-confidence_level/2., 0.5+confidence_level/2.]),
        axis=0
        )
    cum_predict_interval_upper = cum_confidence_bounds[0]
    cum_predict_interval_lower = cum_confidence_bounds[1]

    # Evaluate the causal impact
    impact_point = obs_time_series.squeeze() - predict_point
    impact_interval_lower = obs_time_series.squeeze() - predict_interval_upper
    impact_interval_upper = obs_time_series.squeeze() - predict_interval_lower

    cum_timeseries = jnp.cumsum(obs_time_series.squeeze())
    cum_impact_point = cum_timeseries - cum_predict_point
    cum_impact_interval_lower = cum_timeseries - cum_predict_interval_upper
    cum_impact_interval_upper = cum_timeseries - cum_predict_interval_lower

    impact = {'pointwise': (impact_point, (impact_interval_lower, impact_interval_upper)),
              'cumulative': (cum_impact_point, (cum_impact_interval_lower, cum_impact_interval_upper))}

    predict = {'pointwise': predict_point,
               'interval': confidence_bounds}

    return CausalImpact(sts_model, intervention_timepoint, predict, impact, obs_time_series)
