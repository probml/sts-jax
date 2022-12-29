from collections import namedtuple
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from dynamax.types import PRNGKey
from jaxtyping import Array, Float

import sts_jax.structural_time_series.sts_model as sts


class CausalImpact:
    """A wrapper class of helper functions of the causal impact"""

    def __init__(
        self,
        sts_model: sts.StructuralTimeSeries,
        intervention_time: int,
        predict: dict,
        effect: dict,
        summary: dict,
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
    ) -> None:
        """
        Args:
            sts_model: an instance of the StructualTimeSeries.
            intervention_time: time point of the intervention.
            effect: a dictionary containing pointwise effect and cumulative effect returned by
            the function 'causal_impact'
        """
        self.intervention_time = intervention_time
        self.sts_model = sts_model
        self.predict_point = predict["pointwise"]
        self.predict_interval = predict["interval"]
        self.impact_point = effect["pointwise"]
        self.impact_cumulat = effect["cumulative"]
        self.summary = summary
        self.time_series = obs_time_series

    def plot(self) -> None:
        """Plot the effect."""
        x = jnp.arange(self.time_series.shape[0])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), sharex=True, layout="constrained")

        # Plot the original obvervation and the counterfactual predict
        ax1.plot(x, self.time_series, color="black", lw=2, label="Observation")
        ax1.plot(x, self.predict_point, linestyle="dashed", color="blue", lw=2, label="Prediction")
        ax1.fill_between(x, self.predict_interval[0], self.predict_interval[1], color="blue", alpha=0.2)
        ax1.axvline(x=self.intervention_time, linestyle="dashed", color="gray", lw=2)
        ax1.set_title("Original time series")

        # Plot the pointwise causal impact
        ax2.plot(x, self.impact_point[0], linestyle="dashed", color="blue")
        ax2.fill_between(x, self.impact_point[1][0], self.impact_point[1][1], color="blue", alpha=0.2)
        ax2.axvline(x=self.intervention_time, linestyle="dashed", color="gray", lw=2)
        ax2.set_title("Poinwise causal impact")

        # Plot the cumulative causal impact
        ax3.plot(x, self.impact_cumulat[0], linestyle="dashed", color="blue")
        ax3.fill_between(x, self.impact_cumulat[1][0], self.impact_cumulat[1][1], color="blue", alpha=0.2)
        ax3.axvline(x=self.intervention_time, linestyle="dashed", color="gray", lw=2)
        ax3.set_title("Cumulative causal impact")

        return fig, ax1, ax2, ax3

    def print_summary(self) -> None:
        """Print the summary of the inferred effect as a table."""
        # Number of columns for each column of the table to be printed
        ncol1, ncol2, ncol3 = 25, 20, 20
        ci_level = self.summary["confidence_level"]

        # Summary statistics of the post-intervention observation
        actual = self.summary["actual"]
        f_actual = f"{'Actual': <{ncol1}}" f"{actual.average: ^{ncol2}.2f}" f"{actual.cumulative: ^{ncol3}.2f}\n"

        # Summary statistics of the post-intervention prediction
        pred = self.summary["pred"]
        pred_sd = self.summary["pred_sd"]
        pred_l = self.summary["pred_lower"]
        pred_r = self.summary["pred_upper"]
        f_pred = (
            f"{'Prediction (s.d.)': <{ncol1}}"
            f'{f"{pred.average:.2f} ({pred_sd.average:.2f})": ^{ncol2}}'
            f'{f"{pred.cumulative:.2f} ({pred_sd.cumulative:.2f})": ^{ncol3}}'
        )
        f_pred_ci = (
            f'{f"{ci_level:.0%} CI": <{ncol1}}'
            f'{f"[{pred_l.average:.2f}, {pred_r.average:.2f}]": ^{ncol2}}'
            f'{f"[{pred_l.cumulative:.2f}, {pred_r.cumulative:.2f}]": ^{ncol3}}'
        )

        # Summary statistics of the absolute post-invervention effect
        abs_e = self.summary["abs_effect"]
        abs_sd = self.summary["abs_effect_sd"]
        abs_l = self.summary["abs_effect_lower"]
        abs_r = self.summary["abs_effect_upper"]
        f_abs = (
            f"{'Absolute effect (s.d.)': <{ncol1}}"
            f'{f"{abs_e.average:.2f} ({abs_sd.average:.2f})": ^{ncol2}}'
            f'{f"{abs_e.cumulative:.2f} ({abs_sd.cumulative:.2f})": ^{ncol3}}'
        )
        f_abs_ci = (
            f'{f"{ci_level:.0%} CI": <{ncol1}}'
            f'{f"[{abs_l.average:.2f}, {abs_r.average:.2f}]": ^{ncol2}}'
            f'{f"[{abs_l.cumulative:.2f}, {abs_r.cumulative:.2f}]": ^{ncol3}}'
        )

        # Summary statistics of the relative post-intervention effect
        rel_e = self.summary["rel_effect"]
        rel_sd = self.summary["rel_effect_sd"]
        rel_l = self.summary["rel_effect_lower"]
        rel_r = self.summary["rel_effect_upper"]
        f_rel = (
            f"{'Relative effect (s.d.)': <{ncol1}}"
            f'{f"{rel_e.average:.2%} ({rel_sd.average:.2%})": ^{ncol2}}'
            f'{f"{rel_e.cumulative:.2%} ({rel_sd.cumulative:.2%})": ^{ncol3}}'
        )
        f_rel_ci = (
            f'{f"{ci_level:.0%} CI": <{ncol1}}'
            f'{f"[{rel_l.average:.2%}, {rel_r.average:.2%}]": ^{ncol2}}'
            f'{f"[{rel_l.cumulative:.2%}, {rel_r.cumulative:.2%}]": ^{ncol3}}'
        )

        # Format all statistics
        summary_stats = (
            f"Posterior inference of the causal impact:\n"
            f"\n"
            f"{'': <{ncol1}}{'Average': ^{ncol2}}{'Cumulative': ^{ncol3}}\n"
            f"{f_actual}\n"
            f"{f_pred}\n{f_pred_ci}\n"
            f"\n"
            f"{f_abs}\n{f_abs_ci}\n"
            f"\n"
            f"{f_rel}\n{f_rel_ci}\n"
            f"\n"
            f"Posterior tail-area probability p: {self.summary['tail_prob']:.4f}\n"
            f"Posterior prob of a causal effect: {1-self.summary['tail_prob']:.2%}\n"
        )
        print(summary_stats)


def causal_impact(
    obs_time_series: Float[Array, "num_timesteps dim_obs"],
    intervention_timepoint: int,
    obs_distribution: str = "Gaussian",
    covariates: Optional[Float[Array, "num_timesteps dim_obs"]] = None,
    sts_model: sts.StructuralTimeSeries = None,
    confidence_level: Float = 0.95,
    key: PRNGKey = jr.PRNGKey(0),
    num_samples: int = 200,
) -> CausalImpact:
    r"""Inferring the causal impact of an intervention on a time series via the structural
        time series (STS) model.

    Args:
        obs_time_series: observed time series.
        intervention_time_point: the time point when the intervention took place.
        obs_distribution: distribution family of the observation, can be either 'Gaussian' or
            'Poisson'
        covariates: covariates of the regression component of the STS model.
        sts_model: an instance of StructuralTimeSeries, if not given, an STS model with a local
            linear latent component is used by default. If covariates is not None, a linear
            regression term will also be added to the default STS model.
        confidence_level: confidence level of the prediction interval.
        num_samples: number of samples used to estimate the prediction mean and prediction
            interval.

    Returns:
        An instance of CausalImpact.
    """

    assert obs_distribution in ["Gaussian", "Poisson"]
    if sts_model is not None:
        assert obs_distribution == sts_model.obs_distribution

    prob_lower, prob_upper = 0.5 - confidence_level / 2.0, 0.5 + confidence_level / 2.0
    key1, key2, key3 = jr.split(key, 3)
    num_timesteps, dim_obs = obs_time_series.shape

    # Split the data into pre-intervention period and post-intervention period
    time_series_pre = obs_time_series[:intervention_timepoint]
    time_series_pos = obs_time_series[intervention_timepoint:]

    if covariates is not None:
        dim_covariates = covariates.shape[-1]
        covariates_pre = covariates[:intervention_timepoint]
        covariates_pos = covariates[intervention_timepoint:]
    else:
        covariates_pre = covariates_pos = None

    # Construct the default STS model with only one local linear trend latent component.
    if sts_model is None:
        local_linear_trend = sts.LocalLinearTrend()
        components = [local_linear_trend]
        # Add one linear regression component if covariates is not None.
        if covariates is not None:
            linear_regression = sts.LinearRegression(dim_covariates=dim_covariates)
            components.append(linear_regression)
        sts_model = sts.StructuralTimeSeries(components, obs_time_series, covariates, obs_distribution)

    # Fit the STS model, sample from the past and forecast.
    # Model fitting
    params_posterior_samples, _ = sts_model.fit_hmc(num_samples, time_series_pre, covariates=covariates_pre, key=key1)
    # Sample observations from the posterior predictive sample given paramters of the STS model.
    posterior_sample_means, posterior_samples = sts_model.posterior_sample(
        params_posterior_samples, time_series_pre, covariates_pre, key=key2
    )
    # Forecast by sampling observations from the predictive distribution in the future.
    forecast_means, forecast_samples = sts_model.forecast(
        params_posterior_samples, time_series_pre, time_series_pos.shape[0], 100, covariates_pre, covariates_pos, key3
    )
    forecast_means = forecast_means.mean(axis=1)
    forecast_samples = forecast_samples.mean(axis=1)

    predict_means = jnp.concatenate((posterior_sample_means, forecast_means), axis=1).squeeze()
    predict_observations = jnp.concatenate((posterior_samples, forecast_samples), axis=1).squeeze()

    confidence_bounds = jnp.quantile(predict_observations, jnp.array([prob_lower, prob_upper]), axis=0)
    predict_point = predict_means.mean(axis=0)
    predict_interval_upper = confidence_bounds[0]
    predict_interval_lower = confidence_bounds[1]

    cum_predict_point = jnp.cumsum(predict_point)
    cum_confidence_bounds = jnp.quantile(
        predict_observations.cumsum(axis=1), jnp.array([prob_lower, prob_upper]), axis=0
    )
    cum_predict_interval_upper = cum_confidence_bounds[0]
    cum_predict_interval_lower = cum_confidence_bounds[1]

    # Evaluate the causal impact
    impact_point = obs_time_series.squeeze() - predict_point
    impact_interval_lower = obs_time_series.squeeze() - predict_interval_upper
    impact_interval_upper = obs_time_series.squeeze() - predict_interval_lower

    cum_obs = jnp.cumsum(obs_time_series.squeeze())
    cum_impact_point = cum_obs - cum_predict_point
    cum_impact_interval_lower = cum_obs - cum_predict_interval_upper
    cum_impact_interval_upper = cum_obs - cum_predict_interval_lower

    impact = {
        "pointwise": (impact_point, (impact_interval_lower, impact_interval_upper)),
        "cumulative": (cum_impact_point, (cum_impact_interval_lower, cum_impact_interval_upper)),
    }

    predict = {"pointwise": predict_point, "interval": confidence_bounds}

    summary = dict()
    summary["confidence_level"] = confidence_level
    Stats = namedtuple("Stats", ["average", "cumulative"])

    # Summary statistics of the post-intervention observation
    summary["actual"] = Stats(average=time_series_pos.mean(), cumulative=time_series_pos.sum())

    # Summary statistics of the post-intervention prediction
    summary["pred"] = Stats(average=forecast_means.mean(axis=0).mean(), cumulative=forecast_means.mean(axis=0).sum())
    summary["pred_lower"] = Stats(
        average=jnp.quantile(forecast_samples.mean(axis=1), prob_lower),
        cumulative=jnp.quantile(forecast_samples.sum(axis=1), prob_lower),
    )
    summary["pred_upper"] = Stats(
        average=jnp.quantile(forecast_samples.mean(axis=1), prob_upper),
        cumulative=jnp.quantile(forecast_samples.sum(axis=1), prob_upper),
    )
    summary["pred_sd"] = Stats(
        average=jnp.std(forecast_samples.mean(axis=1)), cumulative=jnp.std(forecast_samples.sum(axis=1))
    )

    # Summary statistics of the absolute post-invervention effect
    effect_means = time_series_pos - forecast_means
    effects = time_series_pos - forecast_samples
    summary["abs_effect"] = Stats(average=effect_means.mean(axis=0).mean(), cumulative=effect_means.mean(axis=0).sum())
    summary["abs_effect_lower"] = Stats(
        average=jnp.quantile(effects.mean(axis=1), prob_lower), cumulative=jnp.quantile(effects.sum(axis=1), prob_lower)
    )
    summary["abs_effect_upper"] = Stats(
        average=jnp.quantile(effects.mean(axis=1), prob_upper), cumulative=jnp.quantile(effects.sum(axis=1), prob_upper)
    )
    summary["abs_effect_sd"] = Stats(average=jnp.std(effects.mean(axis=1)), cumulative=jnp.std(effects.sum(axis=1)))

    # Summary statistics of the relative post-intervention effect
    rel_effect_means_sum = effect_means.sum(axis=1) / forecast_means.sum(axis=1)
    rel_effects_sum = effects.sum(axis=1) / forecast_samples.sum(axis=1)
    summary["rel_effect"] = Stats(average=rel_effect_means_sum.mean(), cumulative=rel_effect_means_sum.mean())
    summary["rel_effect_lower"] = Stats(
        average=jnp.quantile(rel_effects_sum, prob_lower), cumulative=jnp.quantile(rel_effects_sum, prob_lower)
    )
    summary["rel_effect_upper"] = Stats(
        average=jnp.quantile(rel_effects_sum, prob_upper), cumulative=jnp.quantile(rel_effects_sum, prob_upper)
    )
    summary["rel_effect_sd"] = Stats(average=jnp.std(rel_effects_sum), cumulative=jnp.std(rel_effects_sum))

    # Add one-sided tail-area probability of overall impact
    effects_sum = effects.sum(axis=1)
    p_tail = float(1 + min((effects_sum >= 0).sum(), (effects_sum <= 0).sum())) / (1 + effects_sum.shape[0])
    summary["tail_prob"] = p_tail

    return CausalImpact(sts_model, intervention_timepoint, predict, impact, summary, obs_time_series)
