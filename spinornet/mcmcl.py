# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# This file may have been modified by Spinornet authors.
# All Spinornet modifications are Copyright 2025 Spinornet authors.

import jax.numpy as jnp
import jax
from jax import lax
from spinornet import spins_jax
from spinornet import distance
from spinornet.network import logdet_matmul

vdistance = jax.vmap(distance.enforce_pbc, (None, None, 0))

# NOTE:
# I checked that doing grad of the real part (of logdet) is same as grad from value_and_grad of the slogdet.


def mh_accept(x1, x2, feats, feats2, lp_1, lp_2, ratio, key, num_accepts, ages):
    """Given state, proposal, and probabilities, execute MH accept/reject step."""
    key, subkey = jax.random.split(key)

    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    cond = ratio > rnd
    x_new = jnp.where(cond[..., None], x2, x1)

    lp_new = jnp.where(cond, lp_2, lp_1)
    ages_new = jnp.where(cond, 0, ages+1)
    num_accepts += jnp.sum(cond)
    return x_new, feats, key, lp_new, num_accepts, ages_new


def limdrift(g: jnp.array, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector, ndim is 3.

      cutoff: the maximum magnitude

    Returns:
      The vector with the cut off applied.
    """
    g_shape = g.shape
    g = g.reshape([-1, 3])
    tot = jnp.linalg.norm(g, axis=-1)
    normalize = jnp.clip(tot, a_min=cutoff)
    g = cutoff * g / normalize[:, None]
    g = g.reshape(g_shape)
    return g


def mh_update_pos(
    params,
    f,
    pos,
    feats,
    key,
    lp_1,
    num_accepts,
    ages,
    stddev=0.02,
    sigma=0.03,
    pbc=False,
    lattice=None,
    reciplattice=None,
    is2d=False

):
    """Performs one Lanvegin step using an all-electron move.

    Args:
      params: Wavefunction parameters.
      f: val_and_grad of batch_slogdet.
      pos: Initial MCMC configurations (batched).
      feats: Initial MCMC walker spins (batched).
      key: RNG state.
      lp_1: log probability of f evaluated at x1 given parameters params.
      num_accepts: Number of MH move proposals accepted.
      ages: ages of the walkers, i.e. number of steps since last move.
      stddev: width of Gaussian move proposal.
      sigma: width of spin move proposal. Not used.
      pbc: whether to apply periodic boundary conditions.
      lattice: lattice vectors for PBC.
      reciplattice: reciprocal lattice vectors for PBC.
      is2d: whether the system is 2D (i.e. z coordinate is always zero).

    Returns:
      (x, feats, key, lp, num_accepts, ages), where:
        x: Updated MCMC configurations.
        feats: MCMC walker spins.
        key: RNG state.
        lp: log probability of f evaluated at x.
        num_accepts: update running total of number of accepted MH moves.
        ages: updated ages of the walkers.
    """
    key, subkey = jax.random.split(key)
    x1 = pos

    _, grad = f(params, x1, feats)

    grad = limdrift(grad, cutoff=0.1)

    gauss = stddev * jax.random.normal(subkey, shape=x1.shape)
    if is2d:
        gauss = gauss.reshape(gauss.shape[0], -1, 3)
        gauss = gauss.at[:, :, 2].set(0).reshape(gauss.shape[0], -1)

    x2 = x1 + gauss + stddev**2 * grad  # proposal

    if pbc:
        x2, _ = vdistance(lattice, reciplattice, x2)

    lpsi_2, new_grad = f(params, x2, feats)

    new_grad = limdrift(new_grad, cutoff=0.1)

    forward = jnp.sum(gauss**2, axis=-1)
    backward = jnp.sum((gauss + stddev**2 * (grad + new_grad))**2, axis=-1)

    lp_2 = 2.0 * lpsi_2
    lp_move = lp_2 - 1/2/stddev**2 * (backward - forward)
    ratio = lp_move - lp_1
    key, subkey = jax.random.split(key)
    x_new, _, key, p_move, num_accepts_move, ages_new = mh_accept(
        x1, x2, feats, feats, lp_1, lp_move, ratio, subkey, num_accepts, ages)

    return x_new, feats, key, p_move, num_accepts_move, ages_new


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   pbc=False,
                   lattice=None,
                   reciplattice=None,
                   is2d=False
                   ):
    """Creates the MCMC step function.

    Args:
      batch_network: val_and_grad function, signature (params, x, ft), which evaluates the slogdet and gradient. 
      Inputs and outputs are batched.
      batch_per_device: Batch size per device.
      steps: Number of MCMC moves to attempt in a single call to the MCMC step
        function.
      pbc: whether to apply periodic boundary conditions.
      lattice: lattice vectors for PBC.
      reciplattice: reciprocal lattice vectors for PBC.
      is2d: whether the system is 2D (i.e. z coordinate is always zero).

    Returns:
      Callable which performs the set of MCMC steps.
    """

    inner_fun = mh_update_pos

    @jax.jit
    def mcmc_step(params, pos, feats, key, width, sigma, ages):
        """Performs a set of MCMC steps.

        Args:
          params: parameters to pass to the network.
          pos: Initial MCMC configurations (batched).
          feats: Initial MCMC walker spins (batched).
          key: RNG state.
          width: standard deviation to use in the move proposal.
          sigma: width of spin move proposal. Not used.
          ages: Initial ages of the walkers, i.e. number of steps since last move.

        Returns:
          (pos, feats, pmove, ages), where pos is the updated MCMC configurations, feats the
          updated spin configurations, pmove the average probability a move was accepted, 
          ages the updated ages.
        """

        def step_fn(i, x):
            return inner_fun(
                params,
                batch_network,
                *x,
                stddev=width,
                sigma=sigma,
                pbc=pbc,
                lattice=lattice,
                reciplattice=reciplattice,
                is2d=is2d
            )

        nsteps = steps
        # second arg is gradient.
        logprob, _ = batch_network(
            params, pos, feats
        )
        logprob = 2 * logprob
        new_data, new_feats, key, _, num_accepts, ages_new = lax.fori_loop(
            0, nsteps, step_fn, (pos, feats, key, logprob, 0.0, ages)
        )

        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)

        if is2d:
            new_data = new_data.reshape(new_data.shape[0], -1, 3)
            new_data = new_data.at[:, :, 2].set(
                0).reshape(new_data.shape[0], -1)
        return new_data, new_feats, pmove, ages_new

    return mcmc_step
