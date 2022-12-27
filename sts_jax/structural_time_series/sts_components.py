from abc import ABC, abstractmethod
from collections import OrderedDict

import jax.numpy as jnp
import jax.scipy as jsp
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.parameters import ParameterProperties
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import InverseWishart as IW
from dynamax.utils.distributions import MatrixNormalPrecision as MNP
from jax import lax
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiag as MVNDiag,
)
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)
from tensorflow_probability.substrates.jax.distributions import Uniform


class ParamsSTSComponent(OrderedDict):
    """A :class: 'OrderdedDict' with each item being an instance of :class: 'jax.DeviceArray'."""

    pass


class ParamsSTS(OrderedDict):
    """A :class: 'OrderdedDict' with each item being an instance of :class: 'OrderedDict'."""

    pass


class ParamPropertiesSTS(OrderedDict):
    """A :class: 'OrderdedDict' with each item being an instance of :class: 'OrderedDict',
    having the same pytree structure with 'ParamsSTS'.
    """

    pass


class ParamPriorsSTS(OrderedDict):
    """A :class: 'OrderdedDict' with each item being an instance of :class: 'OrderedDict',
    having the same pytree structure with 'ParamsSTS'.
    """

    pass


# helper function
def _initial_statistics(initial_prior, initial_mean, initial_cov):
    if initial_prior is not None:
        return initial_prior.mean(), initial_prior.covariance()
    else:
        return initial_mean, initial_cov


def _set_prior(input_prior, default_prior):
    return default_prior if input_prior is None else input_prior


#########################
#  Abstract Components  #
#########################


class STSComponent(ABC):
    r"""A base class for latent component of structural time series (STS) models.

    **Abstract Methods**

    A latent component of the STS model that inherits from 'STSComponent' must implement
    a few key functions and properties:

    * :meth: 'initialize_params' initializes parameters of the latent component,
        given the initial value and scale of steps of the observed time series.
    * :meth: 'get_trans_mat' returns the transition matrix, $F[t]$, of the latent component
        at time step $t$.
    * :meth: 'get_trans_cov' returns the nonsingular covariance matrix $Q[t]$ of the latent
        component at time step $t$.
    * :attr: 'obs_mat' returns the observation (emission) matrix $H$ for the latent component.
    * :attr: 'cov_select_mat' returns the selecting matrix $R$ that expands the nonsingular
        covariance matrix $Q[t]$ in each time step into a (possibly singular) convarince
        matrix of shape (dim_state, dim_state).
    * :attr: 'name' returns the unique name of the latent component.
    * :attr: 'dim_obs' returns the dimension of the observation in each step of the observed
        time series.
    * :attr: 'initial_distribution' returns the initial_distribution of the initial latent
        state of the component, which is an instance of the class of
        MultivariateNormalFullCovariance
        from tensorflow_probability.substrates.jax.distributions.
    * :attr: 'params' returns parameters of the component, which is an instance of OrderedDict,
        forming a pytree structure of Jax.
    * :attr: 'param_props' returns parameter properties of each item in 'params'.
        param_props has the same pytree structure with 'params', and each leaf is an instance
        of dynamax.parameters.ParameterProperties, which specifies constrainer of
        each parameter and whether that parameter is trainable.
    * :attr: 'param_priors' returns prior distribution of each item in 'params'.
        param_priors has the same pytree structure with 'params', and each leaf is an instance
        of tfd.Distribution.
    """

    def __init__(self, name: str, dim_obs: int = 1) -> None:
        self.name = name
        self.dim_obs = dim_obs
        self.initial_distribution = None

        self.params = OrderedDict()
        self.param_props = OrderedDict()
        self.param_priors = OrderedDict()

    @abstractmethod
    def initialize_params(self, obs_initial: Float[Array, "dim_obs"], obs_scale: Float[Array, "dim_obs"]) -> None:
        r"""Initialize parameters of the component given statistics of the observed time series.

        Args:
            obs_initial: the first observation in the observed time series $z_0$.
            obs_scale: vector of standard deviations of each dimension of the observed time series.

        Returns:
            No returns. Update self.params and self.initial_distributions directly.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trans_mat(self, params: ParamsSTSComponent, t: int) -> Float[Array, "dim_state dim_state"]:
        r"""Compute the transition matrix, $F[t]$, of the latent component at time step $t$.

        Args:
            params: parameters of the latent component, having the same tree structure with
            self.params.
            t: time point at which the transition matrix is to be evaluted.

        Returns:
            transition matrix, $F[t]$, of the latent component at time step $t$
        """
        raise NotImplementedError

    @abstractmethod
    def get_trans_cov(self, params: ParamsSTSComponent, t: int) -> Float[Array, "rank_state rank_state"]:
        r"""Compute the nonsingular covariance matrix, $Q[t]$, of the latent component at
            time step $t$.

        Args:
            params: parameters of the latent component, having the same tree structure with
            self.params.
            t: time point at which the transition matrix is to be evaluted.

        Returns:
            nonsingular covariance matrix, $Q[t]$, of the latent component at time step $t$
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_mat(self) -> Float[Array, "dim_obs dim_state"]:
        r"""Returns the observation (emission) matrix $H$ for the latent component."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_select_mat(self) -> Float[Array, "dim_state rank_state"]:
        r"""Returns the selecting matrix $R$ that expands the nonsingular covariance matrix
            $Q[t]$ in each time step into a (possibly singular) convarince matrix of shape
            (dim_state, dim_state).

        Returns:
            selecting matrix $R$ of shape (dim_state, rank_state)
        """
        raise NotImplementedError


class STSRegression(ABC):
    r"""A base class for regression component of structural time series (STS) models.

    The regression component is not treated as a latent component of the STS model.
    Instead, the value of the regression model in each time step is added to the observation
    model without adding random noise. So the regression model is not necessarily a linear
    regression model, any model is valid, as long as the output of the model has same dimension
    with the observed time series.

    **Abstract Methods**

    Models that inherit from `STSRegression` must implement a few key functions and properties:

    * :meth: 'initialize_params' initializes parameters of the regression model,
        given covariates and the observed time series.
    * :attr: 'name' returns the unique name of the regression component.
    * :attr: 'dim_obs' returns the dimension of the observation in each step of the observed
        time series. This equals the dimension of the output of the regression model.
    * :attr: 'params' returns parameters of the regression function, which is an instance of
        OrderedDict, forming a pytree structure of Jax.
    * :attr: 'param_props' returns parameter properties of each item in 'params'.
        param_props has the same pytree structure with 'params', and each leaf is an instance
        of dynamax.parameters.ParameterProperties, which specifies constrainer of
        each parameter and whether that parameter is trainable.
    * :attr: 'param_priors' returns prior distribution of each item in 'params'.
        param_priors has the same pytree structure with 'params', and each leaf is an instance
        of tfd.Distribution.
    """

    def __init__(self, name: str, dim_obs: int = 1) -> None:
        self.name = name
        self.dim_obs = dim_obs

        self.params = OrderedDict()
        self.param_props = OrderedDict()
        self.param_priors = OrderedDict()

    @abstractmethod
    def initialize_params(
        self,
        covariates: Float[Array, "num_timesteps dim_covariates"],
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
    ) -> None:
        r"""Initialize parameters of the regression model by minimizing certain loss function,
        given the series of covariates and the observed time series.

        Args:
            covariates: series of covariates of the regression function.
            obs_time_series: observed time series.

        Returns:
            No returns. Update self.params directly.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reg_value(
        self, params: ParamsSTSComponent, covariates: Float[Array, "num_timesteps dim_covariates"]
    ) -> Float[Array, "num_timesteps dim_obs"]:
        r"""Returns the sequence of values of the regression model evaluted at the
            given parameters and the sequence of covariates.

        Args:
            params: parameters on which the regression model is to be evalueated.
            covariates: sequence of covariates at which the regression model is to be evaluated.

        Raises:
            values: the sequence of values of the regression model.
        """
        raise NotImplementedError


#########################
#  Concrete Components  #
#########################


class LocalLinearTrend(STSComponent):
    r"""The local linear trend component of the structual time series (STS) model

    The latent state has two parts $[level, slope]$, having dimension 2 * dim_obs.
    The dynamics is:
    $$level[t+1] = level[t] + slope[t] + \matcal{N}(0, cov_level)$$
    $$slope[t+1] = slope[t] + \mathcal{N}(0, cov_slope)$$

    In the case $dim_obs = 1$, the transition matrix $F$ and the observation matrix $H$ are
    $$
    F = \begin{bmatrix}
         1 & 1 \\
         0 & 1
        \end{bmatrix},
    \qquad
    H = [ 1, 0 ].
    $$
    """

    def __init__(
        self,
        dim_obs: int = 1,
        name: str = "local_linear_trend",
        level_cov_prior: tfd.Distribution = None,
        slope_cov_prior: tfd.Distribution = None,
        initial_level_prior: tfd.MultivariateNormalFullCovariance = None,
        initial_slope_prior: tfd.MultivariateNormalFullCovariance = None,
        cov_constrainer: tfb.Bijector = None,
    ) -> None:
        super().__init__(name=name, dim_obs=dim_obs)
        self.level_cov_pri = level_cov_prior
        self.slope_cov_pri = slope_cov_prior
        self.initial_level_pri = initial_level_prior
        self.initial_slope_pri = initial_slope_prior
        self.cov_constrain = RealToPSDBijector() if cov_constrainer is None else cov_constrainer

        self.initial_distribution = MVN(jnp.zeros(2 * dim_obs), jnp.eye(2 * dim_obs))

        self.param_props["cov_level"] = ParameterProperties(trainable=True, constrainer=self.cov_constrain)
        self.param_priors["cov_level"] = _set_prior(level_cov_prior, IW(df=dim_obs, scale=jnp.eye(dim_obs)))
        self.params["cov_level"] = self.param_priors["cov_level"].mode()

        self.param_props["cov_slope"] = ParameterProperties(trainable=True, constrainer=self.cov_constrain)
        self.param_priors["cov_slope"] = _set_prior(slope_cov_prior, IW(df=dim_obs, scale=jnp.eye(dim_obs)))
        self.params["cov_slope"] = self.param_priors["cov_slope"].mode()

        # The local linear trend component has a fixed transition matrix.
        self._trans_mat = jnp.kron(jnp.array([[1, 1], [0, 1]]), jnp.eye(dim_obs))

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.array([1, 0]), jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.eye(2 * dim_obs)

    def initialize_params(self, obs_initial: Float[Array, "dim_obs"], obs_scale: Float[Array, "dim_obs"]) -> None:
        dim_obs = len(obs_initial)

        # Initialize the distribution of the initial state.
        initial_level_mean, initial_level_cov = _initial_statistics(
            self.initial_level_pri, obs_initial, jnp.diag(obs_scale**2)
        )
        initial_slope_mean, initial_slope_cov = _initial_statistics(
            self.initial_slope_pri, jnp.zeros(dim_obs), jnp.diag(obs_scale)
        )
        initial_mean = jnp.concatenate((initial_level_mean, initial_slope_mean))
        initial_cov = jsp.linalg.block_diag(initial_level_cov, initial_slope_cov)
        # initial_cov = jnp.kron(jnp.eye(2), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors["cov_level"] = _set_prior(
            self.level_cov_pri, IW(df=dim_obs, scale=1e-3 * obs_scale**2 * jnp.eye(dim_obs))
        )
        self.params["cov_level"] = self.param_priors["cov_level"].mode()
        self.param_priors["cov_slope"] = _set_prior(
            self.slope_cov_pri, IW(df=dim_obs, scale=1e-3 * obs_scale**2 * jnp.eye(dim_obs))
        )
        self.params["cov_slope"] = self.param_priors["cov_slope"].mode()

    def get_trans_mat(self, params: ParamsSTSComponent, t: int) -> Float[Array, "2*dim_obs 2*dim_obs"]:
        return self._trans_mat

    def get_trans_cov(self, params: ParamsSTSComponent, t: int) -> Float[Array, "2*dim_obs 2*dim_obs"]:
        _shape = params["cov_level"].shape
        return jnp.block([[params["cov_level"], jnp.zeros(_shape)], [jnp.zeros(_shape), params["cov_slope"]]])

    @property
    def obs_mat(self) -> Float[Array, "dim_obs 2*dim_obs"]:
        return self._obs_mat

    @property
    def cov_select_mat(self) -> Float[Array, "2*dim_obs 2*dim_obs"]:
        return self._cov_select_mat


class Autoregressive(STSComponent):
    r"""The autoregressive (AR) latent component of the structural time series (STS) model.

    The autoregressive model of order $p$, i.e., AR(p) is defined as

    $$z_t = \sum_{j=1}^{p} w_j z_{t-j} + \epsilon_t$$

    where

    * $w_j$'s are coefficients of the autoregression. It is required that $|w_j| < 1 $
        to make sure that the autoregressive model is stationary.
    * $\epsilon_t$ is the disturbance that follows a Gaussian distribution with mean 0
        and covariance $cov_level$.

    There are multiple ways to formulate the AR(p) component in the STS model. We set the
    state vector of the component to be the history of the latent state, with the dimension
    $p*dim_obs$.

    If $dim_obs=1$ and assume $p=3$, the transition matrix and the observation matrix is
    $$
    F = \begin{bmatrix}
         w_1 & w_2 & w_3 \\
          1  &  0  &  0  \\
          0  &  1  &  0
        \end{bmatrix},
    \qquad
    H = [1, 0, 0]
    $$
    """

    def __init__(
        self,
        order: int,
        dim_obs: int = 1,
        name: str = "ar",
        coefficients_prior: tfd.Distribution = None,
        cov_level_prior: tfd.Distribution = None,
        initial_state_prior: tfd.MultivariateNormalFullCovariance = None,
        coefficient_constrainer: tfb.Bijector = None,
        cov_constrainer: tfb.Bijector = None,
    ) -> None:
        super().__init__(name=name, dim_obs=dim_obs)
        self.coef_pri = coefficients_prior
        self.cov_level_pri = cov_level_prior
        self.initial_state_pri = initial_state_prior
        self.coef_constrain = tfb.Tanh() if coefficient_constrainer is None else coefficient_constrainer
        self.cov_constrain = RealToPSDBijector() if cov_constrainer is None else cov_constrainer

        self.order = order
        self.initial_distribution = MVN(jnp.zeros(order * dim_obs), jnp.eye(order * dim_obs))

        self.param_props["cov_level"] = ParameterProperties(trainable=True, constrainer=self.cov_constrain)
        self.param_priors["cov_level"] = _set_prior(cov_level_prior, IW(df=dim_obs, scale=jnp.eye(dim_obs)))
        self.params["cov_level"] = self.param_priors["cov_level"].mode()

        self.param_props["coef"] = ParameterProperties(trainable=True, constrainer=self.coef_constrain)
        self.param_priors["coef"] = _set_prior(coefficients_prior, MVNDiag(jnp.zeros(order), jnp.ones(order)))
        self.params["coef"] = self.param_priors["coef"].mode()

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.eye(order)[0], jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.eye(order)[:, 0], jnp.eye(dim_obs))

    def initialize_params(self, obs_initial: Float[Array, "dim_obs"], obs_scale: Float[Array, "dim_obs"]) -> None:
        dim_obs = len(obs_initial)

        # Initialize the distribution of the initial state.
        # if self.initial_state_pri is not None:
        #     initial_state_mean = self.initial_state_pri.mean()
        #     initial_state_cov = self.initial_state_pri.covariance()
        # else:
        #     initial_state_mean = obs_initial
        #     initial_state_cov = jnp.diag(obs_scale**2)
        initial_state_mean, initial_state_cov = _initial_statistics(
            self.initial_state_pri, obs_initial, jnp.diag(obs_scale**2)
        )
        initial_mean = jnp.kron(jnp.eye(self.order)[0], initial_state_mean)
        initial_cov = jnp.kron(jnp.eye(self.order), initial_state_cov)
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors["cov_level"] = _set_prior(
            self.cov_level_pri, IW(df=dim_obs, scale=1e-3 * obs_scale**2 * jnp.eye(dim_obs))
        )
        self.params["cov_level"] = self.param_priors["cov_level"].mode()

    def get_trans_mat(self, params: ParamsSTSComponent, t: int) -> Float[Array, "order*dim_obs order*dim_obs"]:
        trans_mat = jnp.vstack((params["coef"], jnp.eye(self.order)[:-1]))
        return jnp.kron(trans_mat, jnp.eye(self.dim_obs))

    def get_trans_cov(self, params: ParamsSTSComponent, t: int) -> Float[Array, "dim_obs dim_obs"]:
        return params["cov_level"]

    @property
    def obs_mat(self) -> Float[Array, "dim_obs order*dim_obs"]:
        return self._obs_mat

    @property
    def cov_select_mat(self) -> Float[Array, "order*dim_obs dim_obs"]:
        return self._cov_select_mat


class SeasonalDummy(STSComponent):
    r"""The (dummy) seasonal component of the structual time series (STS) model

    Since at any step $t$ the seasonal effect has following constraint

    $$sum_{j=1}^{num_seasons} s_{t-j} = 0 $$,

    the seasonal effect (random) of next time step takes the form:

    $$ s_{t+1} = - sum_{j=1}^{num_seasons-1} s_{t+1-j} + w_{t+1}$$

    where

    * $w_{t+1}$ is the stochastic noise of the seasonal effect following a normal distribution
        with mean 0 and covariance $drift_cov$.
    So the latent state corresponding to the seasonal component has dimension
    $(num_seasons - 1) * dim_obs$

    If $dim_obs = 1$, and suppose that $num_seasons = 4$, the transition matrix and
    the observation matrix is
    $$
    F = \begin{bmatrix}
         -1 & -1 & -1 \\
          1 &  0 &  0 \\
          0 &  1 &  0
        \end{bmatrix},
    \qquad
    H = [ 1, 0, 0 ]
    $$

    Args (in addition to 'name' and 'dim_obs'):
        'num_seasons' is the number of seasons.
        'num_steps_per_season' is consecutive steps that each seasonal effect does not change.
            For example, if a STS model has a weekly seasonal effect but the data is measured
            daily, then num_steps_per_season = 7;
            and if a STS model has a daily seasonal effect but the data is measured hourly,
            then num_steps_per_season = 24.
    """

    def __init__(
        self,
        num_seasons: int,
        num_steps_per_season: int = 1,
        dim_obs: int = 1,
        name: str = "seasonal_dummy",
        drift_cov_prior: tfd.Distribution = None,
        initial_effect_prior: tfd.MultivariateNormalFullCovariance = None,
        cov_constrainer: tfb.Bijector = None,
    ) -> None:
        super().__init__(name=name, dim_obs=dim_obs)
        self.drift_cov_pri = drift_cov_prior
        self.initial_effect_pri = initial_effect_prior
        self.cov_constrain = RealToPSDBijector() if cov_constrainer is None else cov_constrainer

        self.num_seasons = num_seasons
        self.steps_per_season = num_steps_per_season

        _c = self.num_seasons - 1
        self.initial_distribution = MVN(jnp.zeros(_c * dim_obs), jnp.eye(_c * dim_obs))

        self.param_props["drift_cov"] = ParameterProperties(trainable=True, constrainer=self.cov_constrain)
        self.param_priors["drift_cov"] = _set_prior(drift_cov_prior, IW(df=dim_obs, scale=jnp.eye(dim_obs)))
        self.params["drift_cov"] = self.param_priors["drift_cov"].mode()

        # The seasonal component has a fixed transition matrix.
        self._trans_mat = jnp.kron(jnp.concatenate((-jnp.ones((1, _c)), jnp.eye(_c)[:-1]), axis=0), jnp.eye(dim_obs))

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.eye(_c)[0], jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.eye(_c)[:, [0]], jnp.eye(dim_obs))

    def initialize_params(self, obs_initial: Float[Array, "dim_obs"], obs_scale: Float[Array, "dim_obs"]) -> None:
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        # if self.initial_effect_pri is not None:
        #     initial_effect_mean = self.initial_effect_pri.mean()
        #     initial_effect_cov = self.initial_effect_pri.covariance()
        # else:
        #     initial_effect_mean = jnp.zeros(dim_obs)
        #     initial_effect_cov = jnp.diag(obs_scale**2)
        initial_effect_mean, initial_effect_cov = _initial_statistics(
            self.initial_effect_pri, jnp.zeros(dim_obs), jnp.diag(obs_scale**2)
        )
        initial_mean = jnp.kron(jnp.eye(self.num_seasons - 1)[0], initial_effect_mean)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons - 1), initial_effect_cov)
        # initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        # initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors["drift_cov"] = _set_prior(
            self.drift_cov_pri, IW(df=dim_obs, scale=1e-3 * obs_scale**2 * jnp.eye(dim_obs))
        )
        self.params["drift_cov"] = self.param_priors["drift_cov"].mode()

    def get_trans_mat(
        self, params: ParamsSTSComponent, t: int
    ) -> Float[Array, "(num_seasons-1)*dim_obs (num_seasons-1)*dim_obs"]:
        return lax.cond(
            t % self.steps_per_season == 0,
            lambda: self._trans_mat,
            lambda: jnp.eye((self.num_seasons - 1) * self.dim_obs),
        )

    def get_trans_cov(self, params: ParamsSTSComponent, t: int) -> Float[Array, "dim_obs dim_obs"]:
        return lax.cond(
            t % self.steps_per_season == 0,
            lambda: jnp.atleast_2d(params["drift_cov"]),
            lambda: jnp.eye(self.dim_obs) * 1e-32,
        )

    @property
    def obs_mat(self) -> Float[Array, "dim_obs (num_seasons-1)*dim_obs"]:
        return self._obs_mat

    @property
    def cov_select_mat(self) -> Float[Array, "(num_seasons-1)*dim_obs dim_obs"]:
        return self._cov_select_mat


class SeasonalTrig(STSComponent):
    r"""The trigonometric seasonal component of the structual time series (STS) model.

    The seasonal effect (random) of next time step takes the form:

    $$\gamma_t = \sum_{j=1}^{\lfloor s/2 \rfloor} \gamma_{j,t}$$

    and the state is update by

    $$\gamma_{j, t+1}  =  \cos(\lambda_j) \gamma_{j,t} + \sin(\lambda_j) \gamma^*_{jt}  + w_{j,t}$$
    $$\gamma^*_{j, t+1} = -\sin(\lambda_j) \gamma_{j,t} + \cos(\lambda_j) \gamma^*_{jt}  + w^*_{j,t}$$

    where
    * $s$ is number of seasons.
    * $j = 1, ..., \lfloor s/2 \rfloor$.
    * $w_{jt}$ and $w^*_{jt}$ are stochastic noises of the seasonal effect following a normal
        distribution with mean zeros and a common covariance $drift_cov$ for all $j$ and $t$.

    The latent state corresponding to the seasonal component has dimension $(s-1) * dim_obs$.
    If $s$ is odd, then $s-1 = 2 * (s-1)/2$, which means thare are $j = 1,...,(s-1)/2$ blocks.
    If $s$ is even, then $s-1 = 2 * (s/2) - 1$, which means there are $j = 1,...(s/2)$ blocks,
    but we remove the last dimension in this case since this part does not play role in the
    observation.

    If $dim_obs = 1$, for $j = \lfloor (s-1)/2 \rfloor$:
    $$
    F_j = \begin{bmatrix}
            \cos(\lambda_j) & \sin(\lambda_j) \\
           -\sin(\lambda_j) & \cos(\lambda_j)
          \end{bmatrix},
    \qquad
    H_j = [ 1, 0 ]
    $$

    Args (in addition to 'name' and 'dim_obs'):
        'num_seasons' is the number of seasons.
        'num_steps_per_season' is consecutive steps that each seasonal effect does not change.
            For example, if a STS model has a weekly seasonal effect but the data is measured
            daily, then num_steps_per_season = 7;
            and if a STS model has a daily seasonal effect but the data is measured hourly,
            then num_steps_per_season = 24.
    """

    def __init__(
        self,
        num_seasons: int,
        num_steps_per_season: int = 1,
        dim_obs: int = 1,
        name: str = "seasonal_trig",
        drift_cov_prior: tfd.Distribution = None,
        initial_effect_prior: tfd.MultivariateNormalFullCovariance = None,
        cov_constrainer: tfb.Bijector = None,
    ) -> None:
        super().__init__(name=name, dim_obs=dim_obs)
        self.drift_cov_pri = drift_cov_prior
        self.initial_effect_pri = initial_effect_prior
        self.cov_constrain = RealToPSDBijector() if cov_constrainer is None else cov_constrainer

        self.num_seasons = num_seasons
        self.steps_per_season = num_steps_per_season

        _c = num_seasons - 1
        self.initial_distribution = MVN(jnp.zeros(_c * dim_obs), jnp.eye(_c * dim_obs))

        self.param_props["drift_cov"] = ParameterProperties(trainable=True, constrainer=self.cov_constrain)
        self.param_priors["drift_cov"] = _set_prior(drift_cov_prior, IW(df=dim_obs, scale=jnp.eye(dim_obs)))
        self.params["drift_cov"] = self.param_priors["drift_cov"].mode()

        # The seasonal component has a fixed transition matrix.
        num_pairs = int(jnp.floor(num_seasons / 2))
        _trans_mat = jnp.zeros((2 * num_pairs, 2 * num_pairs))
        for j in 1 + jnp.arange(num_pairs):
            lamb_j = (2 * j * jnp.pi) / num_seasons
            block_j = jnp.array([[jnp.cos(lamb_j), jnp.sin(lamb_j)], [-jnp.sin(lamb_j), jnp.cos(lamb_j)]])
            _trans_mat = _trans_mat.at[2 * (j - 1) : 2 * j, 2 * (j - 1) : 2 * j].set(block_j)
        if num_seasons % 2 == 0:
            _trans_mat = _trans_mat[:-1, :-1]
        self._trans_mat = jnp.kron(_trans_mat, jnp.eye(dim_obs))

        # Fixed observation matrix.
        _obs_mat = jnp.tile(jnp.array([1, 0]), num_pairs)
        if num_seasons % 2 == 0:
            _obs_mat = _obs_mat[:-1]
        self._obs_mat = jnp.kron(_obs_mat, jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.eye(_c * dim_obs)

    def initialize_params(self, obs_initial: Float[Array, "dim_obs"], obs_scale: Float[Array, "dim_obs"]) -> None:
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        # if self.initial_effect_pri is not None:
        #     initial_effect_mean = self.initial_effect_pri.mean()
        #     initial_effect_cov = self.initial_effect_pri.covariance()
        # else:
        #     initial_effect_mean = jnp.zeros(dim_obs)
        #     initial_effect_cov = jnp.diag(obs_scale**2)
        initial_effect_mean, initial_effect_cov = _initial_statistics(
            self.initial_effect_pri, jnp.zeros(dim_obs), jnp.diag(obs_scale**2)
        )
        initial_mean = jnp.kron(jnp.eye(self.num_seasons - 1)[0], initial_effect_mean)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons - 1), initial_effect_cov)
        # initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        # initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors["drift_cov"] = _set_prior(
            self.drift_cov_pri, IW(df=dim_obs, scale=1e-3 * obs_scale**2 * jnp.eye(dim_obs))
        )
        self.params["drift_cov"] = self.param_priors["drift_cov"].mode()

    def get_trans_mat(
        self, params: ParamsSTSComponent, t: int
    ) -> Float[Array, "(num_seasons-1)*dim_obs (num_seasons-1)*dim_obs"]:
        return lax.cond(
            t % self.steps_per_season == 0,
            lambda: self._trans_mat,
            lambda: jnp.eye((self.num_seasons - 1) * self.dim_obs),
        )

    def get_trans_cov(
        self, params: ParamsSTSComponent, t: int
    ) -> Float[Array, "(num_seasons-1)*dim_obs (num_seasons-1)*dim_obs"]:
        return lax.cond(
            t % self.steps_per_season == 0,
            lambda: jnp.kron(jnp.eye(self.num_seasons - 1), params["drift_cov"]),
            lambda: jnp.eye((self.num_seasons - 1) * self.dim_obs) * 1e-32,
        )

    @property
    def obs_mat(self) -> Float[Array, "dim_obs (num_seassons-1)*dim_obs"]:
        return self._obs_mat

    @property
    def cov_select_mat(self) -> Float[Array, "(num_seasons-1)*dim_obs (num_seasons-1)*dim_obs"]:
        return self._cov_select_mat


class Cycle(STSComponent):
    r"""The cycle component of the structural time series model.

    The cycle effect (random) of next time step takes the form:

    $$\gamma_t = \cos(freq) + \sin(freq)$$

    and the state is updated by

    $$\gamma_{t+1}   =  \cos(freq) \gamma_t + \sin(freq) \gamma^*_t  + w_t$$
    $$\gamma^*_{t+1} = -\sin(freq) \gamma_t + \cos(freq) \gamma^*_t  + w^*_t$$

    where

    * $w_t$, and $w^*_t$ are stochastic noises of the cycle effect following
        a normal distribution with mean zeros and a common covariance $drift_cov$ for all $t$.

    The latent state corresponding to the cycle component has dimension $2 * dim_obs$.

    If $dim_obs = 1$, the transition matrix and the observation matrix is:
    $$
    F = damp * \begin{bmatrix}
                \cos(freq) & \sin(freq) \\
               -\sin(freq) & \cos(freq)
               \end{bmatrix},
    \qquad
    H = [ 1, 0 ].
    $$

    where

    * $damp$ is the damping factor, and $0 < damp <1$.
    * $freq$ is the frequency factor, and $0 < freq < 2\pi$,
        therefore the period of cycle is $2\pi/freq$.
    """

    def __init__(
        self,
        dim_obs: int = 1,
        name: str = "cycle",
        damping_factor_prior: tfd.Distribution = None,
        frequency_prior: tfd.Distribution = None,
        drift_cov_prior: tfd.Distribution = None,
        initial_effect_prior: tfd.MultivariateNormalFullCovariance = None,
        cov_constrainer: tfb.Bijector = None,
    ) -> None:
        super().__init__(name=name, dim_obs=dim_obs)
        self.damp_pri = damping_factor_prior
        self.frequency_pri = frequency_prior
        self.drift_cov_pri = drift_cov_prior
        self.initial_effect_pri = initial_effect_prior
        self.cov_constrain = RealToPSDBijector() if cov_constrainer is None else cov_constrainer

        self.initial_distribution = MVN(jnp.zeros(2 * dim_obs), jnp.eye(2 * dim_obs))

        # Parameters of the component
        self.param_props["damp"] = ParameterProperties(trainable=True, constrainer=tfb.Sigmoid())
        self.param_priors["damp"] = _set_prior(damping_factor_prior, Uniform(low=0.0, high=1.0))
        self.params["damp"] = self.param_priors["damp"].mode()

        self.param_props["frequency"] = ParameterProperties(
            trainable=True, constrainer=tfb.Sigmoid(low=0.0, high=2 * jnp.pi)
        )
        self.param_priors["frequency"] = _set_prior(frequency_prior, Uniform(low=0.0, high=2 * jnp.pi))
        self.params["frequency"] = self.param_priors["frequency"].mode()

        self.param_props["drift_cov"] = ParameterProperties(trainable=True, constrainer=self.cov_constrain)
        self.param_priors["drift_cov"] = _set_prior(drift_cov_prior, IW(df=dim_obs, scale=jnp.eye(dim_obs)))
        self.params["drift_cov"] = self.param_priors["drift_cov"].mode()

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.array([1, 0]), jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.array([[1], [0]]), jnp.eye(dim_obs))

    def initialize_params(self, obs_initial: Float[Array, "dim_obs"], obs_scale: Float[Array, "dim_obs"]) -> None:
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        # if self.initial_effect_pri is not None:
        #     initial_effect_mean = self.initial_effect_pri.mean()
        #     initial_effect_cov = self.initial_effect_pri.covariance()
        # else:
        #     initial_effect_mean = jnp.zeros(dim_obs)
        #     initial_effect_cov = jnp.diag(obs_scale**2)
        initial_effect_mean, initial_effect_cov = _initial_statistics(
            self.initial_effect_pri, jnp.zeros(dim_obs), jnp.diag(obs_scale**2)
        )
        initial_mean = jnp.kron(jnp.eye(self.num_seasons - 1)[0], initial_effect_mean)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons - 1), initial_effect_cov)
        # initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        # initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors["drift_cov"] = _set_prior(
            self.drift_cov_pri, IW(df=dim_obs, scale=1e-3 * obs_scale**2 * jnp.eye(dim_obs))
        )
        self.params["drift_cov"] = self.param_priors["drift_cov"].mode()

    def get_trans_mat(self, params: ParamsSTSComponent, t: int) -> Float[Array, "2*dim_obs 2*dim_obs"]:
        freq = params["frequency"]
        damp = params["damp"]
        _trans_mat = jnp.array([[jnp.cos(freq), jnp.sin(freq)], [-jnp.sin(freq), jnp.cos(freq)]])
        trans_mat = damp * _trans_mat
        return jnp.kron(trans_mat, jnp.eye(self.dim_obs))

    def get_trans_cov(self, params: ParamsSTSComponent, t: int) -> Float[Array, "dim_obs dim_obs"]:
        return params["drift_cov"]

    @property
    def obs_mat(self) -> Float[Array, "dim_obs 2*dim_obs"]:
        return self._obs_mat

    @property
    def cov_select_mat(self) -> Float[Array, "2*dim_obs dim_obs"]:
        return self._cov_select_mat


class LinearRegression(STSRegression):
    r"""The linear regression component of the structural time series (STS) model.

    The formula of the linear regression component is

    $$y_t = W x_t,$$

    where

    * $u_t$ is the model inputs at time step $t$.
        $u_t = covariates[t]$ is no bias term is to be added, and
        $u_t = [covariates[t], 1]$ is the bias term is a bias term is to be added to the model.
    * $W$ is the coefficient matrix of shape $(dim_obs, dim_covariates)$ if no bias term, and
        $(dim_obs, dim_covariates + 1)$ is a bias term is to be added.
    """

    def __init__(
        self,
        dim_covariates: int,
        add_bias: bool = True,
        dim_obs: int = 1,
        name: str = "linear_regression",
        weights_prior: tfd.Distribution = None,
    ) -> None:
        super().__init__(name=name, dim_obs=dim_obs)
        self.weights_pri = weights_prior
        self.add_bias = add_bias

        dim_inputs = dim_covariates + 1 if add_bias else dim_covariates

        self.param_props["weights"] = ParameterProperties(trainable=True, constrainer=tfb.Identity())
        self.param_priors["weights"] = _set_prior(
            weights_prior,
            MNP(
                loc=jnp.zeros((dim_obs, dim_inputs)), row_covariance=jnp.eye(dim_obs), col_precision=jnp.eye(dim_inputs)
            ),
        )
        self.params["weights"] = jnp.zeros((dim_obs, dim_inputs))

    def initialize_params(
        self,
        covariates: Float[Array, "num_timesteps dim_covariates"],
        obs_time_series: Float[Array, "num_timesteps dim_obs"],
    ) -> None:
        if self.add_bias:
            inputs = jnp.concatenate((covariates, jnp.ones((covariates.shape[0], 1))), axis=1)
        W = jnp.linalg.solve(inputs.T @ inputs, inputs.T @ obs_time_series).T
        self.params["weights"] = W

    def get_reg_value(
        self, params: ParamsSTSComponent, covariates: Float[Array, "num_timesteps dim_covariates"]
    ) -> Float[Array, "num_timesteps dim_obs"]:
        if self.add_bias:
            inputs = jnp.concatenate((covariates, jnp.ones((covariates.shape[0], 1))), axis=1)
            return inputs @ params["weights"].T
        else:
            return covariates @ params["weights"].T
