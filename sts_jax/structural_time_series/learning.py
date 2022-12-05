import blackjax
from collections import OrderedDict
from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit, lax, vmap, value_and_grad
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
from jax.tree_util import tree_map, tree_leaves
import jax.scipy.stats.norm as norm
import optax
from dynamax.parameters import to_unconstrained, from_unconstrained, log_det_jac_constrain
from dynamax.types import PRNGKey
from dynamax.utils.utils import ensure_array_has_batch_dim, pytree_stack
from .sts_ssm import StructuralTimeSeriesSSM
from .sts_components import ParamsSTS, ParamPropertiesSTS
from typing import Optional, Tuple


def fit_vi(
    model: StructuralTimeSeriesSSM,
    initial_params: ParamsSTS,
    param_props: ParamPropertiesSTS,
    num_samples: int,
    emissions: Float[Array, "num_timesteps dim_obs"],
    inputs: Optional[Float[Array, "num_timesteps dim_inputs"]]=None,
    optimizer: optax.GradientTransformation=optax.adam(1e-1),
    K: int=1,
    key: PRNGKey=jr.PRNGKey(0),
    num_step_iters: int=50
) -> Tuple[ParamsSTS, Float[Array, "num_samples"]]:
    """
    ADVI approximate the posterior distribtuion p of unconstrained global parameters
    with factorized multivatriate normal distribution:
    q = \prod_{k=1}^{K} q_k(mu_k, sigma_k),
    where K is dimension of p.

    The hyper-parameters of q to be optimized over are (mu_k, log_sigma_k))_{k=1}^{K}.

    The trick of reparameterization is employed to reduce the variance of SGD,
    which is achieved by written KL(q || p) as expectation over standard normal distribution
    so a sample from q is obstained by
    s = z * exp(log_sigma_k) + mu_k,
    where z is a sample from the standard multivarate normal distribtion.

    Args:
        sample_size (int): number of samples to be returned from the fitted approxiamtion q.
        M (int): number of fixed samples from q used in evaluation of ELBO.

    Returns:
        Samples from the approximate posterior q
    """
    key1, key2 = jr.split(key, 2)
    # Make sure the emissions and covariates have batch dimensions
    batch_emissions = ensure_array_has_batch_dim(emissions, model.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, model.inputs_shape)

    initial_unc_params = to_unconstrained(initial_params, param_props)

    @jit
    def unnorm_log_pos(_unc_params):
        params = from_unconstrained(_unc_params, param_props)
        log_det_jac = log_det_jac_constrain(params, param_props)
        log_pri = model.log_prior(params) + log_det_jac
        batch_lls = vmap(partial(model.marginal_log_prob, params))(batch_emissions, batch_inputs)
        lp = batch_lls.sum() + log_pri
        return lp

    @jit
    def elbo(vi_hyper, key):
        """Evaluate negative ELBO at fixed sample from the approximate distribution q.
        """
        keys = iter(jr.split(key, 10))
        # Turn VI parameters and fixed noises into samples of unconstrained parameters of q.
        unc_params = tree_map(lambda mu, ls: mu + jnp.exp(ls) * jr.normal(next(keys), ls.shape).sum(),
                              vi_hyper['mu'], vi_hyper['log_sig'])
        log_probs = unnorm_log_pos(unc_params)
        log_q = jnp.array(tree_leaves(tree_map(lambda x, *p: norm.logpdf(x, p[0], jnp.exp(p[1])).sum(),
                                               unc_params, vi_hyper['mu'], vi_hyper['log_sig']))).sum()
        return log_probs - log_q

    loss_fn = lambda vi_hyp, key: -jnp.mean(vmap(partial(elbo, vi_hyp))(jr.split(key, K)))

    # Fit
    curr_vi_mus = initial_unc_params
    curr_vi_log_sigs = tree_map(lambda x: jnp.zeros(x.shape), initial_unc_params)
    curr_vi_hyper = OrderedDict()
    curr_vi_hyper['mu'] = curr_vi_mus
    curr_vi_hyper['log_sig'] = curr_vi_log_sigs

    # Optimize
    opt_state = optimizer.init(curr_vi_hyper)
    loss_grad_fn = value_and_grad(loss_fn)

    def train_step(carry, key):
        vi_hyp, opt_state = carry
        loss, grads = loss_grad_fn(vi_hyp, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        vi_hyp = optax.apply_updates(vi_hyp, updates)
        return (vi_hyp, opt_state), loss
    # Run the optimizer
    initial_carry = (curr_vi_hyper, opt_state)
    (vi_hyp_fitted, opt_state), losses = lax.scan(
        train_step, initial_carry, jr.split(key1, num_step_iters))

    # Sample from the learned approximate posterior q
    vi_sample = lambda key: from_unconstrained(
        tree_map(lambda mu, s: mu + jnp.exp(s)*jr.normal(key, s.shape),
                 vi_hyp_fitted['mu'], vi_hyp_fitted['log_sig']),
        param_props)
    samples = vmap(vi_sample)(jr.split(key2, num_samples))

    return samples, losses


def fit_hmc(
    model: StructuralTimeSeriesSSM,
    initial_params: ParamsSTS,
    param_props: ParamPropertiesSTS,
    num_samples: int,
    emissions: Float[Array, "num_timesteps dim_obs"],
    inputs: Optional[Float[Array, "num_timesteps dim_inputs"]]=None,
    key: PRNGKey=jr.PRNGKey(0),
    warmup_steps: int=100,
    verbose: bool=True
) -> Tuple[ParamsSTS, Float[Array, "num_samples"]]:
    """Sample parameters of the model using HMC.
    """
    # Make sure the emissions and covariates have batch dimensions
    batch_emissions = ensure_array_has_batch_dim(emissions, model.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, model.inputs_shape)

    initial_unc_params = to_unconstrained(initial_params, param_props)

    # The log likelihood that the HMC samples from
    def unnorm_log_pos(_unc_params):
        params = from_unconstrained(_unc_params, param_props)
        log_det_jac = log_det_jac_constrain(params, param_props)
        log_pri = model.log_prior(params) + log_det_jac
        batch_lls = vmap(partial(model.marginal_log_prob, params))(batch_emissions, batch_inputs)
        lp = log_pri + batch_lls.sum()
        return lp

    # Initialize the HMC sampler using window_adaptations
    warmup = blackjax.window_adaptation(blackjax.nuts, unnorm_log_pos, num_steps=warmup_steps,
                                        progress_bar=verbose)
    init_key, key = jr.split(key)
    hmc_initial_state, hmc_kernel, _ = warmup.run(init_key, initial_unc_params)

    @jit
    def hmc_step(hmc_state, step_key):
        next_hmc_state, _ = hmc_kernel(step_key, hmc_state)
        params = from_unconstrained(hmc_state.position, param_props)
        return next_hmc_state, params

    # Start sampling.
    log_probs = []
    samples = []
    hmc_state = hmc_initial_state
    pbar = progress_bar(range(num_samples)) if verbose else range(num_samples)
    for _ in pbar:
        step_key, key = jr.split(key)
        hmc_state, params = hmc_step(hmc_state, step_key)
        log_probs.append(-hmc_state.potential_energy)
        samples.append(params)

    # Combine the samples into a single pytree
    return pytree_stack(samples), jnp.array(log_probs)
