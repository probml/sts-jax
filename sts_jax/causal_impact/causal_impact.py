from collections import namedtuple
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
    def __init__(self,
                 sts_model,
                 intervention_time,
                 predict,
                 causal_impact,
                 summary,
                 observed_timeseries):
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
        self.summary = summary
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

    def print_summary(self):
        # Number of columns for each column of the table to be printed
        ncol1, ncol2, ncol3 = 22, 15, 15
        ci_level = str(95)

        # Summary statistics of the post-intervention observation
        actual = self.summary('actual')
        f_actual = (
            f"{'Actual': <{ncol1}}"
            f"{actual.average: <{ncol2}}"
            f"{actual.cumulative: <{ncol3}}\n"
        )

        # Summary statistics of the post-intervention prediction
        pred = self.summary['pred']
        pred_sd = self.summary['pred_sd']
        pred_l = self.summary['pred_lower']
        pred_r = self.summary['pred_upper']
        f_pred = (
            f"{'Prediction (s.d.)': <{ncol1}}"
            f"{str(pred.average) + '('+str(pred_sd.averate)+')': <{ncol2}}"
            f"{str(pred.cumulative) + '('+str(pred_sd.cumulative)+')': <{ncol3}}"
        )
        f_pred_ci = (
            f"{ci_level + '%  CI': <{ncol1}}"
            f"{'[' + str(pred_l.average) + ',' + str(pred_r.average) + ']': <{ncol2}}"
            f"{'[' + str(pred_l.cumulative) + ',' + str(pred_r.cumulative) + ']': <{ncol3}}"
        )

        # Summary statistics of the absolute post-invervention effect
        abs_e = self.summary['abs_effect']
        abs_sd = self.summary['abs_effect_sd']
        abs_l = self.summary['abs_effect_lower']
        abs_r = self.summary['abs_effect_upper']
        f_abs = (
            f"{'Absolute effect (s.d.)': <{ncol1}}"
            f"{str(abs_e.average) + '('+str(abs_sd.averate)+')': <{ncol2}}"
            f"{str(abs_e.cumulative) + '('+str(abs_sd.cumulative)+')': <{ncol3}}"
        )
        f_abs_ci = (
            f"{ci_level+'%  CI': <{ncol1}}"
            f"{'[' + str(abs_l.average) + ',' + str(abs_r.average) + ']': <{ncol2}}"
            f"{'[' + str(abs_l.cumulative) + ',' + str(abs_r.cumulative) + ']': <{ncol3}}"
        )

        # Summary statistics of the relative post-intervention effect
        rel_e = self.summary['rel_effect']
        rel_sd = self.summary['rel_effect_sd']
        rel_l = self.summary['rel_effect_lower']
        rel_r = self.summary['rel_effect_upper']
        f_rel = (
            f"{'Relative effect (s.d.)': <{ncol1}}"
            f"{str(rel_e.average) + '('+str(rel_sd.averate)+')': <{ncol2}}"
            f"{str(rel_e.cumulative) + '('+str(rel_sd.cumulative)+')': <{ncol3}}"
        )
        f_rel_ci = (
            f"{ci_level+'% . CI': <{ncol1}}"
            f"{'[' + str(rel_l.average) + ',' + str(rel_r.average) + ']': <{ncol2}}"
            f"{'[' + str(rel_l.cumulative) + ',' + str(rel_r.cumulative) + ']': <{ncol3}}"
        )

        # Format all statistics
        summary_stats = (
            f"Posterior inference of the causal impact:\n"
            f"\n"
            f"{'': <{ncol1}} {'Average': <{ncol2}} {'Cumulative': <{ncol3}}\n"
            f"{f_actual}\n"
            f"{f_pred}\n {f_pred_ci}\n"
            f"\n"
            f"{f_abs}\n  {f_abs_ci}\n"
            f"\n"
            f"{f_rel}\n  {f_rel_ci}\n"
            f"\n"
            f"Posterior tail-area probability p: \n"
            f"Posterior prob of a causal effect: \n"
            )

        print(summary_stats)


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
        params_posterior_samples, _ = sts_model.fit_hmc(num_samples, time_series_pre,
                                                        covariates=covariates_pre, key=key1)
        # Sample from the past and forecast
        samples_pre = sts_model.posterior_sample(
            params_posterior_samples, time_series_pre, covariates_pre, key=key2)
        samples_pos = sts_model.forecast(
            params_posterior_samples, time_series_pre, time_series_pos.shape[0],
            covariates_pre, covariates_pos, key3)
    else:
        # Model fitting
        params_posterior_samples, _ = sts_model.fit_hmc(num_samples, time_series_pre, key=key1)
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

    summary = dict()
    Stats = namedtuple("Stats", ["average", "cumulative"])

    # Summary statistics of the post-intervention observation
    summary['actual'] = Stats(average=time_series_pos.mean(),
                              cumulative=time_series_pos.sum())

    # Summary statistics of the post-intervention prediction
    summary['pred'] = Stats(average=predict_point[intervention_timepoint:].mean(),
                            cumulative=predict_point[intervention_timepoint:].sum())
    summary['pred_lower'] = Stats(average=None,
                                  cumulative=None)
    summary['pred_upper'] = Stats(average=None,
                                  cumulative=None)
    summary['pred_sd'] = Stats(average=None,
                               cumulative=None)

    # Summary statistics of the absolute post-invervention effect
    summary['abs_effect'] = Stats(average=None,
                                  cumulative=None)
    summary['abs_effect_lower'] = Stats(average=None,
                                        cumulative=None)
    summary['abs_effect_upper'] = Stats(average=None,
                                        cumulative=None)
    summary['abs_effect_sd'] = Stats(average=None,
                                     cumulative=None)

    # Summary statistics of the relative post-intervention effect
    summary['rel_effect'] = Stats(average=None,
                                  cumulative=None)
    summary['rel_effect_lower'] = Stats(average=None,
                                        cumulative=None)
    summary['rel_effect_upper'] = Stats(average=None,
                                        cumulative=None)
    summary['rel_effect_sd'] = Stats(average=None,
                                     cumulative=None)

    return CausalImpact(
        sts_model, intervention_timepoint, predict, impact, summary, obs_time_series)
