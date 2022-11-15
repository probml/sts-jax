from abc import ABC, abstractmethod
from collections import OrderedDict
from dynamax.distributions import (
    InverseWishart as IW, MatrixNormalPrecision as MNP)
from dynamax.parameters import ParameterProperties as Prop
from dynamax.utils import PSDToRealBijector
from jax import lax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    MultivariateNormalDiag as MVNDiag,
    Uniform)
import tensorflow_probability.substrates.jax.bijectors as tfb


RealToPSD = tfb.Invert(PSDToRealBijector)


#########################
#  Abstract Components  #
#########################


class STSComponent(ABC):
    """Meta class of latent component of structural time series (STS) models.

    A latent component of the STS model has following attributes:

    name (string): name of the latend component.
    dim_obs (int): dimension of the observation in each step of the observed time series.
    initial_distribution (MVN): an instance of MultivariateNormalFullCovariance,
        specifies the distribution for the inital state.
    params (OrderedDict): parameters of the component need to be learned in model fitting.
    param_props (OrderedDict): properties of each item in 'params'.
        Each item is an instance of ParameterProperties, which specifies constrainer
        of the parameter and whether the parameter is trainable.
    param_priors (OrderedDict): prior distributions for each item in 'params'.
    """

    def __init__(self, name, dim_obs=1):
        self.name = name
        self.dim_obs = dim_obs
        self.initial_distribution = None

        self.param_props = OrderedDict()
        self.param_priors = OrderedDict()
        self.params = OrderedDict()

    @abstractmethod
    def initialize_params(self, obs_initial, obs_scale):
        """Initialize parameters in self.params given the scale of the observed time series.

        Args:
            obs_initial (self.dim_obs,): the first observation in the observed time series.
            obs_scale (self.dim_obs,): scale of the observed time series.

        Returns:
            No returns. Change self.params and self.initial_distributions directly.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trans_mat(self, params, t):
        """Compute the transition matrix at step t of the latent dynamics.

        Args:
            params (OrderedDict): parameters based on which the transition matrix
                is to be evalueated. Has the same tree structure with self.params.
            t (int): time steps

        Returns:
            trans_mat (dim_of_state, dim_of_state): transition matrix at step t
        """
        raise NotImplementedError

    @abstractmethod
    def get_trans_cov(self, params, t):
        """Nonsingular covariance matrix"""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_mat(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_select_mat(self):
        """Select matrix that makes the covariance matrix singular"""
        raise NotImplementedError


class STSRegression(ABC):
    """Meta class of regression component of structural time series (STS) models.

    The regression component has the same dimension as the observed time series.
    This component will appear in the observation (emission) model in the state space model
    form of the STS, since it does not have the random term, in contrast to latent components.

    A regression component of the STS model has following attributes:

    name (string): name of the regression component.
    dim_obs (int): dimension of the observation in each step of the observed time series.
        This is also the output of the regression model.
    params (OrderedDict): parameters of the function need to be learned in model fitting.
    param_props (OrderedDict): properties of each item in 'params'.
        Each item is an instance of ParameterProperties, which specifies constrainer
        of the parameter and whether the parameter is trainable.
    param_priors (OrderedDict): prior distributions for each item in 'params'.
    """

    def __init__(self, name, dim_obs=1):
        self.name = name
        self.dim_obs = dim_obs

        self.param_props = OrderedDict()
        self.param_priors = OrderedDict()
        self.params = OrderedDict()

    @abstractmethod
    def initialize(self, covariates, obs_time_series):
        """Initialize parameters of the regression function by minimising the least square error,
        given the series of covariates and the observed time series.

        Args:
            covariates (len(obs_time_series), dim_covariates): series of inputs
                of the regression function.
            obs_time_series: observed time series.
        Returns:
            No returns. Change self.params and self.initial_distributions directly.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, params, covariates):
        """Compute the fitted value of the regression model at one time step
            given parameters and covariates.

        Args:
            params (OrderedDict): parameters based on which the regression model
                is to be evalueated. Has the same tree structure with self.params.
            covariates (dim_cov, ): covariates at which the regression model is to be evaluated

        Raises:
            fitted (dim_obs,): the fitted value of the regression model.
        """
        raise NotImplementedError


#########################
#  Concrete Components  #
#########################


class LocalLinearTrend(STSComponent):
    """The local linear trend component of the structual time series (STS) model

    The latent state is [level, slope], having dimension 2 * dim_obs. The dynamics is

        level[t+1] = level[t] + slope[t] + N(0, cov_level)
        slope[t+1] = slope[t] + N(0, cov_slope)

    In the case dim_obs = 1:

        trans_mat = | 1, 1 |    obs_mat = [ 1, 0 ]
                    | 0, 1 |,
    """

    def __init__(self,
                 level_cov_prior=None,
                 slope_cov_prior=None,
                 initial_level_prior=None,
                 initial_slope_prior=None,
                 dim_obs=1,
                 name='local_linear_trend'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.initial_distribution = MVN(jnp.zeros(2*dim_obs), jnp.eye(2*dim_obs))

        self.param_props['cov_level'] = Prop(trainable=True, constrainer=RealToPSD)
        self.param_priors['cov_level'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['cov_level'] = self.param_priors['cov_level'].mode()

        self.param_props['cov_slope'] = Prop(trainable=True, constrainer=RealToPSD)
        self.param_priors['cov_slope'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['cov_slope'] = self.param_priors['cov_slope'].mode()

        # The local linear trend component has a fixed transition matrix.
        self._trans_mat = jnp.kron(jnp.array([[1, 1], [0, 1]]), jnp.eye(dim_obs))

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.array([1, 0]), jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.eye(2*dim_obs)

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.concatenate((obs_initial, jnp.zeros(dim_obs)))
        initial_cov = jnp.kron(jnp.eye(2), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors['cov_level'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['cov_level'] = self.param_priors['cov_level'].mode()
        self.param_priors['cov_slope'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['cov_slope'] = self.param_priors['cov_slope'].mode()

    def get_trans_mat(self, params, t):
        return self._trans_mat

    def get_trans_cov(self, params, t):
        _shape = params['cov_level'].shape
        return jnp.block([[params['cov_level'], jnp.zeros(_shape)],
                          [jnp.zeros(_shape), params['cov_slope']]])

    @property
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class Autoregressive(STSComponent):
    """The autoregressive component of the structural time series (STS) model.

    Args (in addition to name and dim_obs):
        p (int): the autoregressive order
    """
    def __init__(self,
                 order,
                 coefficients_prior=None,
                 level_cov_prior=None,
                 initial_state_prior=None,
                 coefficient_constraining_bijector=None,
                 dim_obs=1,
                 name='ar'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.order = order
        self.initial_distribution = MVN(jnp.zeros(order*dim_obs), jnp.eye(order*dim_obs))

        self.param_props['cov_level'] = Prop(trainable=True, constrainer=RealToPSD)
        self.param_priors['cov_level'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['cov_level'] = self.param_priors['cov_level'].mode()

        self.param_props['coef'] = Prop(trainable=True, constrainer=tfb.Tanh())
        self.param_priors['coef'] = MVNDiag(jnp.zeros(order), jnp.ones(order))
        self.params['coef'] = self.param_priors['coef'].mode()

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.eye(order)[0], jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.eye(order)[:, 0], jnp.eye(dim_obs))

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.kron(jnp.eye(self.order)[0], obs_initial)
        initial_cov = jnp.kron(jnp.eye(self.order), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors['cov_level'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['cov_level'] = self.param_priors['cov_level'].mode()

    def get_trans_mat(self, params, t):
        if self.order == 1:
            trans_mat = params['coef'][:, None]
        else:
            trans_mat = jnp.concatenate((params['coef'][:, None],
                                         jnp.eye(self.order)[:, :-1]), axis=1)
        return jnp.kron(trans_mat, jnp.eye(self.dim_obs))

    def get_trans_cov(self, params, t):
        return params['cov_level']

    @property
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class SeasonalDummy(STSComponent):
    """The (dummy) seasonal component of the structual time series (STS) model

    Since at any step t the seasonal effect has following constraint

        sum_{j=1}^{num_seasons} s_{t-j} = 0,

    the seasonal effect (random) of next time step takes the form:

        s_{t+1} = - sum_{j=1}^{num_seasons-1} s_{t+1-j} + w_{t+1}

    and the last term w_{t+1} is the stochastic noise of the seasonal effect
    following a normal distribution with mean zero and covariance drift_cov.
    So the latent state corresponding to the seasonal component has dimension
    (num_seasons - 1) * dim_obs

    If dim_obs = 1, and suppose num_seasons = 4

                    | -1, -1, -1 |
        trans_mat = |  1,  0   0 |    obs_mat = [ 1, 0, 0 ]
                    |  0,  1,  0 |,

    Args (in addition to name and dim_obs):
        num_seasons (int): number of seasons.
        num_steps_per_season: consecutive steps that each seasonal effect does not change.
            For example, if a STS model has a weekly seasonal effect but the data is measured
            daily, then num_steps_per_season = 7;
            and if a STS model has a daily seasonal effect but the data is measured hourly,
            then num_steps_per_season = 24.
    """

    def __init__(self,
                 num_seasons,
                 num_steps_per_season=1,
                 allow_drift=True,
                 drift_cov_prior=None,
                 initial_effect_prior=None,
                 dim_obs=1,
                 name='seasonal_dummy'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.num_seasons = num_seasons
        self.steps_per_season = num_steps_per_season

        _c = self.num_seasons - 1
        self.initial_distribution = MVN(jnp.zeros(_c*dim_obs), jnp.eye(_c*dim_obs))

        self.param_props['drift_cov'] = Prop(trainable=True, constrainer=RealToPSD)
        self.param_priors['drift_cov'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['drift_cov'] = self.param_priors['drift_cov'].mode()

        # The seasonal component has a fixed transition matrix.
        self._trans_mat = jnp.kron(jnp.concatenate((-jnp.ones((1, _c)), jnp.eye(_c)[:-1]), axis=0),
                                   jnp.eye(dim_obs))

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.eye(_c)[0], jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.eye(_c)[:, [0]], jnp.eye(dim_obs))

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors['drift_cov'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['drift_cov'] = self.param_priors['drift_cov'].mode()

    def get_trans_mat(self, params, t):
        return lax.cond(t % self.steps_per_season == 0,
                        lambda: self._trans_mat,
                        lambda: jnp.eye((self.num_seasons-1)*self.dim_obs))

    def get_trans_cov(self, params, t):
        return lax.cond(t % self.steps_per_season == 0,
                        lambda: jnp.atleast_2d(params['drift_cov']),
                        lambda: jnp.eye(self.dim_obs)*1e-32)

    @property
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class SeasonalTrig(STSComponent):
    """The trigonometric seasonal component of the structual time series (STS) model.

    The seasonal effect (random) of next time step takes the form (let s:=num_seasons):

        \gamma_t = \sum_{j=1}^{floor(s/2)} \gamma_{jt}

    where

        \gamma_{j, t+1}  =  cos(\lambda_j) \gamma_{jt} + sin(\lambda_j) \gamma*_{jt}  + w_{jt}
        \gamma*_{j, t+1} = -sin(\lambda_j) \gamma_{jt} + cos(\lambda_j) \gamma*_{jt}  + w*_{jt}

    for j = 1, ..., floor(s/2).
    The last term w_{jt}, w^*_{jt} are stochastic noises of the seasonal effect following
    a normal distribution with mean zeros and a common covariance drift_cov for all j and t.

    The latent state corresponding to the seasonal component has dimension (s-1) * dim_obs.
    If s is odd, then s-1 = 2 * (s-1)/2, which means thare are j = 1,...,(s-1)/2 blocks.
    If s is even, then s-1 = 2 * (s/2) - 1, which means there are j = 1,...(s/2) blocks,
    but we remove the last dimension since this part does not play role in the observation.

    If dim_obs = 1, for j = floor((s-1)/2):

        trans_mat_j = |  cos(\lambda_j), sin(\lambda_j) |    obs_mat_j = [ 1, 0 ]
                      | -sin(\lambda_j), cos(\lambda_j) |,

    Args (in addition to name and dim_obs):
        num_seasons (int): number of seasons.
        num_steps_per_season: consecutive steps that each seasonal effect does not change.
            For example, if a STS model has a weekly seasonal effect but the data is measured
            daily, then num_steps_per_season = 7;
            and if a STS model has a daily seasonal effect but the data is measured hourly,
            then num_steps_per_season = 24.
    """

    def __init__(self,
                 num_seasons,
                 num_steps_per_season=1,
                 allow_drift=True,
                 drift_cov_prior=None,
                 initial_state_prior=None,
                 dim_obs=1,
                 name='seasonal_trig'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.num_seasons = num_seasons
        self.steps_per_season = num_steps_per_season

        _c = num_seasons - 1
        self.initial_distribution = MVN(jnp.zeros(_c*dim_obs), jnp.eye(_c*dim_obs))

        self.param_props['drift_cov'] = Prop(trainable=True, constrainer=RealToPSD)
        self.param_priors['drift_cov'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['drift_cov'] = self.param_priors['drift_cov'].mode()

        # The seasonal component has a fixed transition matrix.
        num_pairs = int(jnp.floor(num_seasons/2))
        _trans_mat = jnp.zeros((2*num_pairs, 2*num_pairs))
        for j in 1 + jnp.arange(num_pairs):
            lamb_j = (2*j * jnp.pi) / num_seasons
            block_j = jnp.array([[jnp.cos(lamb_j), jnp.sin(lamb_j)],
                                 [-jnp.sin(lamb_j), jnp.cos(lamb_j)]])
            _trans_mat = _trans_mat.at[2*(j-1):2*j, 2*(j-1):2*j].set(block_j)
        if num_seasons % 2 == 0:
            _trans_mat = _trans_mat[:-1, :-1]
        self._trans_mat = jnp.kron(_trans_mat, jnp.eye(dim_obs))

        # Fixed observation matrix.
        _obs_mat = jnp.tile(jnp.array([1, 0]), num_pairs)
        if num_seasons % 2 == 0:
            _obs_mat = _obs_mat[:-1]
        self._obs_mat = jnp.kron(_obs_mat, jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.eye(_c*dim_obs)

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors['drift_cov'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['drift_cov'] = self.param_priors['drift_cov'].mode()

    def get_trans_mat(self, params, t):
        return lax.cond(t % self.steps_per_season == 0,
                        lambda: self._trans_mat,
                        lambda: jnp.eye((self.num_seasons-1)*self.dim_obs))

    def get_trans_cov(self, params, t):
        return lax.cond(t % self.steps_per_season == 0,
                        lambda: jnp.kron(jnp.eye(self.num_seasons-1), params['drift_cov']),
                        lambda: jnp.eye((self.num_seasons-1)*self.dim_obs)*1e-32)

    @property
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class Cycle(STSComponent):
    """The cycle component of the structural time series model.

    The cycle effect (random) of next time step takes the form:

        \gamma_t = cos(freq) + sin(freq)

    where

        \gamma_{t+1}  =  cos(freq) \gamma_t + sin(freq) \gamma*_t  + w_t
        \gamma*_{t+1} = -sin(freq) \gamma_t + cos(freq) \gamma*_t  + w*_t

    The last term w_t, w^*_t are stochastic noises of the cycle effect following
    a normal distribution with mean zeros and a common covariance drift_cov for all t.

    The latent state corresponding to the cycle component has dimension 2 * dim_obs.

    If dim_obs = 1:

        trans_mat_j = damp * |  cos(freq), sin(freq) |    obs_mat_j = [ 1, 0 ]
                             | -sin(freq), cos(freq) |,

    damp (scalar): damping factor, 0 < damp <1.
    freq (scalar): frequency factor, 0 < freq < 2\pi,
        therefore the period of cycle is 2\pi/freq.
    """

    def __init__(self,
                 damping_factor_prior=None,
                 frequency_prior=None,
                 drift_cov_prior=None,
                 initial_effect_prior=None,
                 dim_obs=1,
                 name='cycle'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.initial_distribution = MVN(jnp.zeros(2*dim_obs), jnp.eye(2*dim_obs))

        # Parameters of the component
        self.param_props['damp'] = Prop(trainable=True, constrainer=tfb.Sigmoid())
        self.param_priors['damp'] = Uniform(low=0., high=1.)
        self.params['damp'] = self.param_priors['damp'].mode()

        self.param_props['frequency'] = Prop(trainable=True,
                                             constrainer=tfb.Sigmoid(low=0., high=2*jnp.pi))
        self.param_priors['frequency'] = Uniform(low=0., high=2*jnp.pi)
        self.params['frequency'] = self.param_priors['frequency'].mode()

        self.param_props['drift_cov'] = Prop(trainable=True, constrainer=RealToPSD)
        self.param_priors['drift_cov'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['drift_cov'] = self.param_priors['drift_cov'].mode()

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.array([1, 0]), jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.array([[1], [0]]), jnp.eye(dim_obs))

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale**2))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.param_priors['drift_cov'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['drift_cov'] = self.param_priors['drift_cov'].mode()

    def get_trans_mat(self, params, t):
        freq = params['frequency']
        damp = params['damp']
        _trans_mat = jnp.array([[jnp.cos(freq), jnp.sin(freq)],
                                [-jnp.sin(freq), jnp.cos(freq)]])
        trans_mat = damp * _trans_mat
        return jnp.kron(trans_mat, jnp.eye(self.dim_obs))

    def get_trans_cov(self, params, t):
        return params['drift_cov']

    @property
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class LinearRegression(STSRegression):
    """The linear regression component of the structural time series (STS) model.

    The parameter of the linear regression function is the coefficient matrix W.
    The shape of W is (dim_obs, dim_covariates) if no bias term is added, and is
    (dim_obs, dim_covariates + 1) is a bias term is added at the end of the covariates.
    The regression function has the form:

        f(W, X) = W @ X

    where X = covariates if add_bias=False and X = [covariates, 1] if add_bias=True.
    """
    def __init__(self,
                 dim_covariates,
                 add_bias=True,
                 weights_prior=None,
                 dim_obs=1,
                 name='linear_regression'):
        super().__init__(name=name, dim_obs=dim_obs)
        self.add_bias = add_bias

        dim_inputs = dim_covariates + 1 if add_bias else dim_covariates

        self.param_props['weights'] = Prop(trainable=True, constrainer=tfb.Identity())
        self.param_priors['weights'] = MNP(loc=jnp.zeros((dim_inputs, dim_obs)),
                                           row_covariance=jnp.eye(dim_inputs),
                                           col_precision=jnp.eye(dim_obs))
        self.params['weights'] = jnp.zeros((dim_inputs, dim_obs))

    def initialize(self, covariates, obs_time_series):
        if self.add_bias:
            inputs = jnp.concatenate((covariates, jnp.ones((covariates.shape[0], 1))), axis=1)
        W = jnp.linalg.solve(inputs.T @ inputs, inputs.T @ obs_time_series)
        self.params['weights'] = W

    def fit(self, params, covariates):
        if self.add_bias:
            inputs = jnp.concatenate((covariates, jnp.ones((covariates.shape[0], 1))), axis=1)
            return inputs @ params['weights']
        else:
            return covariates @ params['weights']
