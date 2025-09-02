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
vlogdet_matmul = jax.vmap(logdet_matmul, 0)


def mh_accept(x1, x2, feats, feats2, lp_1, lp_2, ratio, key, num_accepts, ages):
    """Given state, proposal, and probabilities, execute MH accept/reject step."""
    key, subkey = jax.random.split(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    cond = ratio > rnd
    x_new = jnp.where(cond[..., None], x2, x1)
    feats_new = jnp.where(cond[..., None, None], feats2, feats)
    lp_new = jnp.where(cond, lp_2, lp_1)
    ages_new = jnp.where(cond, 0, ages+1)
    num_accepts += jnp.sum(cond)
    return x_new, feats_new, key, lp_new, num_accepts, ages_new


def mh_accept_spin(x1, x2, feats, feats2,  lp_1, lp_2, ratio, key, num_accepts, ages):
    """Given state, proposal, and probabilities, execute MH accept/reject step."""
    key, subkey = jax.random.split(key)

    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    cond = ratio > rnd
    feats_new = jnp.where(cond[..., None, None], feats2, feats)

    lp_new = jnp.where(cond, lp_2, lp_1)

    same = jnp.all(feats_new == feats, axis=(1, 2))
    ages_new = jnp.where(same, ages+1, 0)
    num_accepts += jnp.sum(cond)
    return x1, feats_new, key, lp_new, num_accepts, ages_new


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
    is2d=False,
):
    """Performs one Metropolis-Hastings step using an all-electron move.

    Args:
      params: Wavefunction parameters.
      f: Callable with signature f(params, x, ft) which returns the log of the
        wavefunction (i.e. the sqaure root of the log probability of x, ft).
      pos: Initial MCMC configurations (batched).
      feats: Initial MCMC walker spins (batched).
      key: RNG state.
      lp_1: log probability of f evaluated at x1 given parameters params.
      num_accepts: Number of MH move proposals accepted.
      ages: Initial ages of the walkers, i.e. number of steps since last move.
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
        lp: log probability of f evaluated at x, ft.
        num_accepts: update running total of number of accepted MH moves.
        ages: ages of the walkers, i.e. number of steps since last move.
    """
    key, subkey = jax.random.split(key)
    x1 = pos
    propmove = stddev * jax.random.normal(subkey, shape=x1.shape)

    x2 = x1 + propmove  # proposal

    if pbc:
        x2, _ = vdistance(lattice, reciplattice, x2)
    lp_move = 2.0 * f(params, x2, feats)

    ratio = lp_move - lp_1
    x_new, feats_new, key, p_move, num_accepts_move, ages = mh_accept(
        x1, x2, feats, feats, lp_1, lp_move, ratio, key, num_accepts, ages)

    return x_new, feats_new, key, p_move, num_accepts_move, ages


def mh_update_spin(
    params,
    f,
    pos,
    feats,
    key,
    lp_1,
    mats,
    num_accepts,
    ages,
    stddev=0.02,
    sigma=0.03,
):
    """Performs one Metropolis-Hastings step using cycling through electron spin move.

    Args:
      params: Wavefunction parameters.
      f: Not used.
      pos: Initial MCMC configurations (batched).
      feats: Initial MCMC walker spins (batched).
      key: RNG state.
      lp_1: log probability of f evaluated at x1 given parameters params.
      mats: neural network matrix for current walker positions.
      num_accepts: Number of MH move proposals accepted.
      ages: Initial ages of the walkers, i.e. number of steps since last move.
      stddev: width of Gaussian move proposal. Not used.
      sigma: width of spin move proposal.

    Returns:
      (x, feats, key, lp, mats, num_accepts, ages), where:
        x: MCMC configurations.
        feats: Updated MCMC walker spins.
        key: RNG state.
        lp: log probability of f evaluated at x, ft.
        mats: neural network matrix for current walker positions.
        num_accepts: update running total of number of accepted MH moves.
        ages: ages of the walkers, i.e. number of steps since last move.
    """
    key, subkey = jax.random.split(key)

    nelec = feats.shape[1]
    ii = jax.random.permutation(subkey, nelec)

    key, subkey = jax.random.split(key)
    moves = spins_jax.sample_vmf_spins_jax(subkey, feats.shape[:-1], sigma)

    feats2 = spins_jax.rotate_to_jax(moves, feats)

    def step_fn(i, x):
        feats, key, lp_1, num_accepts, ages = x
        featsone2 = feats.at[:, ii[i]].set(feats2[:, ii[i]])
        spinorbitals = jnp.einsum(
            "...es,...eods -> ...deo ", jnp.conj(featsone2), mats)
        _, result = vlogdet_matmul(spinorbitals)
        lp_spin = 2.0 * (result)
        ratio = lp_spin - lp_1
        _, feats_new, key, p, num_accepts_spin, ages = mh_accept(
            pos, pos, feats, featsone2, lp_1, lp_spin, ratio, key, num_accepts, ages)
        return feats_new, key, p, num_accepts_spin, ages

    feats_new, key, p, num_accepts_spin, agesspin = lax.fori_loop(
        0, nelec, step_fn, (feats, key, lp_1, 0.0,
                            jnp.zeros(shape=feats.shape[0]))
    )
    agesspin = jnp.where(agesspin == nelec, ages+1, 0)

    return pos, feats_new, key, p, mats, num_accepts_spin, agesspin


def mh_update_spin_discrete(
    params,
    f,
    pos,
    feats,
    key,
    lp_1,
    mats,
    num_accepts,
    ages,
    stddev=0.02,
    sigma=0.03,
):
    """Performs one Metropolis-Hastings step using cycling through discrete electron spin move.

    Args:
      params: Wavefunction parameters.
      f: Not used.
      pos: Initial MCMC configurations (batched).
      feats: Initial MCMC walker spins (batched).
      key: RNG state.
      lp_1: log probability of f evaluated at x1 given parameters params.
      mats: neural network matrix for current walker positions.
      num_accepts: Number of MH move proposals accepted.
      ages: Initial ages of the walkers, i.e. number of steps since last move.
      stddev: width of Gaussian move proposal. Not used.
      sigma: width of spin move proposal.

    Returns:
      (x, feats, key, lp, mats, num_accepts, ages), where:
        x: MCMC configurations.
        feats: Updated MCMC walker spins.
        key: RNG state.
        lp: log probability of f evaluated at x, ft.
        mats: neural network matrix for current walker positions.
        num_accepts: update running total of number of accepted MH moves.
        ages: ages of the walkers, i.e. number of steps since last move.
    """
    key, subkey = jax.random.split(key)

    nelec = feats.shape[1]
    ii = jax.random.permutation(subkey, nelec)

    def step_fn(i, x):
        feats, key, lp_1, num_accepts, ages = x
        feats2 = feats.at[:, ii[i]].set(jnp.where(feats[:, ii[i]] == jnp.array(
            [1.+0j, 0]), jnp.array([0j, 1.+0j]), jnp.array([1.+0j, 0j])))
        spinorbitals = jnp.einsum(
            "...es,...eods -> ...deo ", jnp.conj(feats2), mats)
        _, result = vlogdet_matmul(spinorbitals)

        lp_spin = 2.0 * (result)
        ratio = lp_spin - lp_1
        _, feats_new, key, p, num_accepts_spin, ages = mh_accept_spin(
            pos, pos, feats, feats2, lp_1, lp_spin, ratio, key, num_accepts, ages)
        return feats_new, key, p, num_accepts_spin, ages

    feats_new, key, p, num_accepts_spin, agesspin = lax.fori_loop(
        0, nelec, step_fn, (feats, key, lp_1, 0.0,
                            jnp.zeros(shape=feats.shape[0]))
    )

    agesspin = jnp.where(agesspin == nelec, ages + 1, 0)

    return pos, feats_new, key, p, mats, num_accepts_spin, agesspin


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   pbc=False,
                   lattice=None,
                   reciplattice=None,
                   is2d=False,
                   ):
    """Creates the MCMC step function.

    Args:
      batch_network: function, signature (params, x, ft), which evaluates the log of
        the wavefunction (square root of the log probability distribution) at x, ft
        given params. Inputs and outputs are batched.
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
          sigma: width of spin move proposal. 
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

        logprob = 2.0 * batch_network(
            params, pos, feats
        )
        new_data, new_feats, key, _, num_accepts, new_ages = lax.fori_loop(
            0, nsteps, step_fn, (pos, feats, key, logprob, 0.0, ages)
        )

        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)

        if is2d:
            new_data = new_data.reshape(new_data.shape[0], -1, 3)
            new_data = new_data.at[:, :, 2].set(
                0).reshape(new_data.shape[0], -1)
        return new_data, new_feats, pmove, new_ages

    return mcmc_step


def make_mcmc_step_spin(batch_mats,
                        batch_per_device,
                        steps=10,
                        ):
    """Creates the MCMC step function.

    Args:
      batch_mats: function, signature (params, x, ft), which evaluates the neural
        network orbitals at x, ft
        given params. Inputs and outputs are batched.
      batch_per_device: Batch size per device.
      steps: Number of MCMC moves to attempt in a single call to the MCMC step
        function.

    Returns:
      Callable which performs the set of MCMC steps.
    """

    inner_fun = mh_update_spin

    @jax.jit
    def mcmc_step(params, pos, feats, key, width, sigma, ages):
        """Performs a set of MCMC steps.

        Args:
          params: parameters to pass to the network.
          pos: Initial MCMC configurations (batched).
          feats: Initial MCMC walker spins (batched).
          key: RNG state.
          width: standard deviation to use in the move proposal.
          sigma: width of spin move proposal. 
          ages: Initial ages of the walkers, i.e. number of steps since last move.

        Returns:
          (pos, feats, pmove, ages), where pos is the updated MCMC configurations, feats the
          updated spin configurations, pmove the average probability a move was accepted, 
          ages the updated ages.
        """

        def step_fn(i, x):
            return inner_fun(
                params,
                batch_mats,
                *x,
                stddev=width,
                sigma=sigma,
            )
        nelec = pos.shape[-1]//3

        nsteps = steps

        mats = batch_mats(params, pos, feats)

        spinorbitals = jnp.einsum(
            "...es,...eods -> ...deo ", jnp.conj(feats), mats)

        _, result = vlogdet_matmul(spinorbitals)

        logprob = 2.0 * (result)

        new_data, new_feats, key, _, _, num_accepts, new_ages = lax.fori_loop(
            0, nsteps, step_fn, (pos, feats, key, logprob, mats, 0.0, ages)
        )

        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device * nelec)

        return new_data, new_feats, pmove, new_ages

    return mcmc_step


def make_mcmc_step_spin_discrete(batch_mats,
                                 batch_per_device,
                                 steps=10,
                                 ):
    """Creates the MCMC step function.

    Args:
      batch_mats: function, signature (params, x, ft), which evaluates the neural
        network orbitals at x, ft
        given params. Inputs and outputs are batched.
      batch_per_device: Batch size per device.
      steps: Number of MCMC moves to attempt in a single call to the MCMC step
        function.

    Returns:
      Callable which performs the set of MCMC steps.
    """

    inner_fun = mh_update_spin_discrete

    @jax.jit
    def mcmc_step(params, pos, feats, key, width, sigma, ages):
        """Performs a set of MCMC steps.

        Args:
          params: parameters to pass to the network.
          pos: Initial MCMC configurations (batched).
          feats: Initial MCMC walker spins (batched).
          key: RNG state.
          width: standard deviation to use in the move proposal.
          sigma: width of spin move proposal. 
          ages: Initial ages of the walkers, i.e. number of steps since last move.

        Returns:
          (pos, feats, pmove, ages), where pos is the updated MCMC configurations, feats the
          updated spin configurations, pmove the average probability a move was accepted, 
          ages the updated ages.
        """

        def step_fn(i, x):
            return inner_fun(
                params,
                batch_mats,
                *x,
                stddev=width,
                sigma=sigma,
            )
        nelec = pos.shape[-1]//3

        nsteps = steps

        mats = batch_mats(params, pos, feats)
        spinorbitals = jnp.einsum(
            "...es,...eods -> ...deo ", jnp.conj(feats), mats)
        _, result = vlogdet_matmul(spinorbitals)

        logprob = 2.0 * (result)

        new_data, new_feats, key, _, _,  num_accepts, new_ages = lax.fori_loop(
            0, nsteps, step_fn, (pos, feats, key, logprob, mats,  0.0, ages)
        )

        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device * nelec)

        return new_data, new_feats, pmove, new_ages

    return mcmc_step
