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

import jax
import jax.numpy as jnp
import numpy as np
import pyscf.pbc.gto
from typing import Sequence
from spinornet.utils import system
from spinornet import spins_jax
from spinornet import distance

venforce_pbc = jax.vmap(distance.enforce_pbc, (None, None, 0))


def init_electrons(
        key,
        cell: Sequence[system.Atom],
        electrons: Sequence[int],
        batch_size: int,
        init_width=0.5,
        latvec=None,
) -> jnp.ndarray:
    """
    Initializes electron positions around each atom.

    :param key: jax key for random
    :param cell: internal representation of simulation cell
    :param electrons: list of up, down electrons
    :param batch_size: batch_size for simulation
    :param init_width: std of gaussian used for initialization
    :param latvec: lattice vector of cell
    :return: jnp.array with shape [Batch_size, N_ele * ndim]
    """
    if sum(atom.charge for atom in cell) != sum(electrons):
        if len(cell) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:

        atomic_spin_configs = [
            (atom.element.nalpha - int((atom.atomic_number - atom.charge) // 2),
             atom.element.nbeta - int((atom.atomic_number - atom.charge) // 2))
            for atom in cell
        ]
        # element.nalpha return the up spin number of the single element, if ecp is used, [nalpha,nbeta] should be reduce
        # with the the core charge which equals atomic_number - atom.charge
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            if atomic_spin_configs[i][0] > 0:
                atomic_spin_configs[i] = nalpha - 1, nbeta + 1

    # Assign each electron to an atom initially.
    electron_positions = []
    for j in range(len(cell)):
        atom_position = jnp.asarray(cell[j].coords)
        electron_positions.append(atom_position)
    electron_positions = jnp.concatenate(electron_positions)
    # Create a batch of configurations with a Gaussian distribution about each
    # atom.
    key, subkey = jax.random.split(key)
    guess = electron_positions + init_width * \
        jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
    # if latvec then do this step.
    replaced_guess, _ = distance.enforce_pbc(
        latvec, jnp.linalg.inv(latvec), guess)
    return replaced_guess


def init_electrons_mol_jax(
    key,
    molecule,
    electrons,
    batch_size: int,
    init_width=0.5,
) -> jnp.ndarray:
    """Initializes electron positions around each atom.

    Args:
      key: JAX RNG state.
      molecule: system.Atom objects making up the molecule.
      electrons: tuple of number of alpha and beta electrons.
      batch_size: total number of MCMC configurations to generate across all
        devices.
      init_width: width of (atom-centred) Gaussian used to generate initial
        electron configurations.

    Returns:
      array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
      positions in the initial MCMC configurations and ndim is the dimensionality
      of the space (i.e. typically 3).
    """
    if sum(atom.charge for atom in molecule) != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:
        atomic_spin_configs = [
            (atom.element.nalpha, atom.element.nbeta) for atom in molecule
        ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            atomic_spin_configs[i] = nbeta, nalpha

    # Assign each electron to an atom initially.
    electron_positions = []
    for i in range(2):
        for j in range(len(molecule)):
            atom_position = jnp.asarray(molecule[j].coords)
            electron_positions.append(
                jnp.tile(atom_position, atomic_spin_configs[j][i]))
    electron_positions = jnp.concatenate(electron_positions)
    # Create a batch of configurations with a Gaussian distribution about each
    # atom.
    key, subkey = jax.random.split(key)
    return (
        electron_positions +
        init_width *
        jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


def init_spins(nelec: int, batch_size: int):
    '''
    :param nelec: total number of electrons
    :return: jnp.array shape(nelec, 2)
    '''
    spinup = jnp.array([[1., 0.]])
    spindown = jnp.array([[0., 1.]])
    if nelec % 2 == 0:
        feats = jnp.concatenate(
            (jnp.tile(spinup, (nelec//2, 1)), jnp.tile(spindown, (nelec//2, 1))))
    else:
        feats = jnp.concatenate(
            (jnp.tile(spinup, (nelec//2 + 1, 1)), jnp.tile(spindown, (nelec//2, 1))))
    return jnp.tile(feats[None, ...], (batch_size, 1, 1))


def init_spins_half(key, datainit, mol, sigma=None):
    batch_size, nelec3 = datainit.shape
    nelec = nelec3 // 3
    feats = init_spins(nelec, batch_size)
    return feats + jnp.array([0j, 0j])


def init_spins_nelec(key, datainit, mol, nelec):
    spinup = jnp.array([[1., 0.]])
    spindown = jnp.array([[0., 1.]])

    batch_size, nelec3 = datainit.shape
    feats = jnp.concatenate(
        (jnp.tile(spinup, (nelec[0], 1)), jnp.tile(spindown, (nelec[1], 1))))
    feats = jnp.tile(feats[None, ...], (batch_size, 1, 1))
    feats = feats + jnp.array([0j, 0j])
    return feats


def init_spins_random(key, datainit, mol, sigma=None):
    batch_size, nelec3 = datainit.shape
    nelec = nelec3//3
    key, subkey = jax.random.split(key)

    z = jax.random.uniform(subkey, shape=(
        batch_size, nelec), minval=-1., maxval=1.)
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey, shape=(
        batch_size, nelec), minval=0., maxval=2*jnp.pi)
    spinor = spins_jax.cylindrical2spinorjax(jnp.stack([z, phi], axis=-1))
    return spinor


def pyscf_to_cell(cell: pyscf.pbc.gto.Cell):
    """
    Converts the pyscf cell to the internal representation.

    :param cell: pyscf.cell object
    :return: internal cell representation
    """
    internal_cell = [system.Atom(cell.atom_symbol(i),
                                 cell.atom_coords()[i],
                                 charge=cell.atom_charges()[i], )
                     for i in range(cell.natm)]
    return internal_cell
