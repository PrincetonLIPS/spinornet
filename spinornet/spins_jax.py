# Copyright 2025 Spinornet authors.

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

import jax.numpy as jnp
import jax
import numpy as np


def cylindrical2spinorjax(zphi):
    out = jnp.ones((*zphi.shape[:-1], 2), dtype=jnp.complex128)
    costh2 = jnp.sqrt((1 + zphi[..., 0]) / 2)
    sinth2 = jnp.sqrt((1 - zphi[..., 0]) / 2)
    out = out.at[..., 0].set(costh2)
    out = out.at[..., 1].set(sinth2 * jnp.exp(1j * zphi[..., 1]))
    return out


def spinor2cartjax(spin):
    r"""
    ..math:: (\cos(\theta/2), e^{i\phi}\sin(\theta/2)) \rightarrow (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)
    """
    # assert np.all(spin[:, 0].imag < 1e-12), np.amax(spin[:, 0].imag)
    # assert np.all(spin[:, 0].imag < 1e-12)
    out = jnp.zeros((*spin.shape[:-1], 3))
    out = out.at[..., 0].set(spin[..., 1].real * 2 * spin[..., 0].real)
    out = out.at[..., 1].set(spin[..., 1].imag * 2 * spin[..., 0].real)
    out = out.at[..., 2].set(2 * spin[..., 0].real ** 2 - 1)
    return out


PAULI_MATRICES = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
)


def propose_random_rotation_spin_move(key, spins, sigma):
    # 1. Choose a random (x, y, z) normalized vector
    spinor = spins
    key, splitkey = jax.random.split(key)
    v = jax.random.normal(splitkey, shape=(*spinor.shape[:-1], 3))
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    # 2. Choose an angle alpha to rotate by
    key, splitkey = jax.random.split(key)
    alpha = jax.random.normal(splitkey, shape=spinor.shape[:-1]) * sigma
    # 3. Generate the rotation operator
    mat = jnp.einsum("...,...i,ijk->...jk", alpha, v, PAULI_MATRICES[1:])
    R = jax.scipy.linalg.expm(-.5j * mat)
    newspins = jnp.einsum("...jk,...k->...j", R, spinor)
    phase_factor = jnp.exp(-1j * jnp.angle(newspins[..., :1]))
    return newspins * phase_factor


def rotate_to_jax(moves, spins):
    spinor = spins
    Rphi = spinor / jnp.abs(spinor)  # Rz to y=0 plane
    Rphi = jnp.where(jnp.abs(spinor) < 1e-8, 1.0, Rphi)
    Y = jnp.array([1, -1j])[:, jnp.newaxis, jnp.newaxis] * \
        PAULI_MATRICES[[0, 2]]
    new_spins = jnp.einsum(
        "...i,ijk,...k,...j->...j",
        jnp.abs(spinor),
        Y,  # rotate about y-axis
        Rphi.conj() * moves,  # rotate phi to y=0 plane
        Rphi,  # rotate back to the original phi
        optimize="greedy",
    )
    phase_factor = jnp.exp(-1j * jnp.angle(new_spins[..., :1]))
    new_spins = new_spins * phase_factor
    return new_spins


def sample_vmf_spins_jax(key, nconf, sigma):
    # Von Mises-Fisher distribution is like an isotropic gaussian on the sphere https://en.wikipedia.org/wiki/Von_Mises–Fisher_distribution
    # 1. Generate random vector around north pole
    # 2. Rotate it to center around desired (theta, phi)
    # 3. Force bias: rotate according to derivative dtheta, dphi
    key, splitkey = jax.random.split(key)
    kappa = 1 / (2 * sigma**2)
    xi = jax.random.uniform(splitkey, shape=nconf)
    z = 1 + 1 / kappa * jnp.log(xi + (1 - xi) * jnp.exp(-2 * kappa))
    key, splitkey = jax.random.split(key)
    phi = jax.random.uniform(splitkey, shape=nconf, minval=0, maxval=2*jnp.pi)
    spinor = cylindrical2spinorjax(jnp.stack([z, phi], axis=-1))
    return spinor


def flip_spinx(feats):
    new_spins = (PAULI_MATRICES[1] @ feats.T).T
    return new_spins


def flip_spiny(feats):
    new_spins = (PAULI_MATRICES[2] @ feats.T).T
    return new_spins
