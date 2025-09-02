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

import haiku as hk
import jax.numpy as jnp
from jax import vmap
from typing import Callable, Sequence, Optional, List
import jax
import numpy as np
from spinornet import scatter
import functools
from collections import namedtuple
from math import prod
from spinornet import spins_jax
from spinornet import curvature_tags_and_blocks
from spinornet import distance
from typing import Any, Union
from spinornet import init_guess


def get_rvec(pos, atompos, ndim=3):
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atompos[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    return ae, ee


class HFOrbital(hk.Module):
    def __init__(self, nelec, name=None):
        super().__init__(name=name)
        self.nelec = nelec

    def __call__(self, pos):
        spin_mos = jnp.ones((sum(self.nelec), sum(self.nelec), 2))
        return spin_mos

    @classmethod
    def from_pyscf(cls, hf):
        """Create the input of the constructor from a :class:`spinornet.hf.SCF`.

        Args:
            hf (spinornet.hf.SCFMol): the hartree-fock object to consider.
        """
        return (hf.nelec,)


class HFOrbitalcol(hk.Module):
    def __init__(self, nelec, name=None):
        super().__init__(name=name)
        self.nelec = nelec

    def __call__(self, pos):
        spin_mos = np.zeros((sum(self.nelec), sum(self.nelec), 2))
        spin_mos[:, 0:self.nelec[0], 0] = 1
        spin_mos[:, self.nelec[0]:, 1] = 1
        return spin_mos

    @classmethod
    def from_pyscf(cls, hf):
        """Create the input of the constructor from a :class:`spinornet.hf.SCF`.

        Args:
            hf (spinornet.hf.SCFMol): the hartree-fock object to consider.
        """
        return (hf.nelec,)


class HFOrbitalHEG2dcol(hk.Module):
    def __init__(self, hartree_fockHEG, name=None):
        super().__init__(name=name)
        self.ks = hartree_fockHEG.klist
        self.nelec = hartree_fockHEG.nelec

        lattice = hartree_fockHEG.lattice[:2, :2]
        self.reciplatvec = jnp.linalg.inv(lattice)
        self.metric = lattice.T @ lattice

    def __call__(self, pos):
        datatest = pos.reshape(-1, 3)[:, :2]
        kdotx = jnp.matmul(datatest, self.ks[:, :2].T)
        wavesre = jnp.cos(kdotx)
        wavesim = jnp.sin(kdotx)

        # collinear env
        spin_mos = jnp.zeros((sum(self.nelec), sum(self.nelec), 2, 2))

        for s in [0, 1]:
            i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]
            spin_mos = spin_mos.at[:, i0:i1, s, 0].set(
                wavesre[:, :self.nelec[s]])
            spin_mos = spin_mos.at[:, i0:i1, s, 1].set(
                wavesim[:, :self.nelec[s]])

        out = spin_mos
        return out[..., 0] + 1.j*out[..., 1]


class HFOrbitalHEG2d(hk.Module):
    def __init__(self, hartree_fockHEG, name=None):
        super().__init__(name=name)
        self.ks = hartree_fockHEG.klist
        self.nelec = hartree_fockHEG.nelec

    def __call__(self, pos):
        datatest = pos.reshape(-1, 3)[:, :2]
        kdotx = jnp.matmul(datatest, self.ks[:, :2].T)
        wavesre = jnp.cos(kdotx)
        wavesim = jnp.sin(kdotx)

        spin_mos = jnp.ones((sum(self.nelec), sum(self.nelec), 2, 1))
        spin_mos2 = jnp.zeros((sum(self.nelec), sum(self.nelec), 2, 1))
        spin_mos = spin_mos.at[:, 0].set(0)
        spin_mos = spin_mos.at[:, self.nelec[0]].set(0)
        spin_mos = jnp.concatenate((spin_mos, spin_mos2), axis=-1)

        for s in [0, 1]:
            i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]

            spin_mos = spin_mos.at[:, i0:i1, s, 0].set(
                wavesre[:, :self.nelec[s]])
            spin_mos = spin_mos.at[:, i0:i1, s, 1].set(
                wavesim[:, :self.nelec[s]])

        out = spin_mos
        return out[..., 0] + 1.j*out[..., 1]


class HFOrbitalHEG2dRb(hk.Module):
    def __init__(self, hartree_fockHEG, lb, name=None):
        super().__init__(name=name)
        self.ks = hartree_fockHEG.klist
        self.nelec = hartree_fockHEG.nelec
        absk = np.linalg.norm(self.ks, axis=1)
        self.alphacoeff = (1j*self.ks[:, 0] + self.ks[:, 1])/absk
        self.alphacoeff = self.alphacoeff.at[0].set(1+1.j)

        self.eindsn = np.argsort((absk**2/2 - lb*absk))
        self.eindsp = np.argsort((absk**2/2 + lb*absk))

        self.einds = jnp.concatenate(
            (self.eindsn[:self.nelec[0]], self.eindsp[:self.nelec[1]]))

        self.alphac = jnp.ones([1, sum(self.nelec), 2], dtype=jnp.complex128)
        keepalpha = jnp.concatenate(
            (-self.alphacoeff[self.eindsn[:self.nelec[0]]], self.alphacoeff[self.eindsp[:self.nelec[1]]]))
        self.alphac = self.alphac.at[:, :, 0].set(keepalpha)

    def __call__(self, pos):
        datatest = pos.reshape(-1, 3)[:, :2]
        kdotx = jnp.matmul(datatest, self.ks[:, :2].T)
        wavesre = jnp.cos(kdotx)[:, self.einds]
        wavesim = jnp.sin(kdotx)[:, self.einds]

        spin_mos = jnp.ones((sum(self.nelec), sum(self.nelec), 2, 2))
        for s in [0, 1]:
            spin_mos = spin_mos.at[:, :, s, 0].set(wavesre)
            spin_mos = spin_mos.at[:, :, s, 1].set(wavesim)

        out = spin_mos
        out = out[..., 0] + 1.j*out[..., 1]
        return out * self.alphac


def isotropic_envelope(ae, env_sigma, env_pi):
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    return jnp.sum(jnp.exp(-jnp.abs(ae * env_sigma)) * env_pi, axis=1)


class spinornet_single(hk.Module):

    def __init__(self, simulation_cell, lattice, klist=None, hidden_size=None,
                 hidden_size_single=256, ndets=1, name=None):
        super().__init__(name=name)
        self.ndets = ndets
        self.init_output_size(simulation_cell)

        self.init_atom_params(simulation_cell)

        self.lattice = lattice
        self.hidden_size_single = hidden_size_single

        self.fcf = hk.Linear(self.output_size)

        self.fc1_s = hk.Linear(self.hidden_size_single)
        self.fc2_s = hk.Linear(self.hidden_size_single)

        edge_index = np.stack((np.meshgrid(np.arange(sum(self.nelec)), np.arange(
            sum(self.nelec)))), -1)[~np.eye(sum(self.nelec), dtype=bool)]
        self.senders = edge_index[:, 0]
        self.receivers = edge_index[:, 1]

        self.klist = klist
        self.init_env_params()

    def init_output_size(self, simulation_cell):
        self.nelec = simulation_cell.nelec
        self.output_size = 2*2*sum(self.nelec)*self.ndets

    def init_atom_params(self, simulation_cell):
        self.atoms = simulation_cell.atom_coords()
        self.natoms = self.atoms.shape[0]

    def init_env_params(self):
        self.envelope_sigma = hk.get_parameter("env_sigma", shape=[
                                               1, self.natoms, self.output_size], dtype=jnp.float64, init=jnp.ones)
        self.envelope_pi = hk.get_parameter("env_pi", shape=[
                                            1, self.natoms, self.output_size], dtype=jnp.float64, init=jnp.ones)

    def preprocess(self, positions: jnp.array):
        ae, ee = get_rvec(positions, self.atoms)
        # removes the self-distance (otherwise need to deal with r=0)
        ee = ee[~np.eye(sum(self.nelec), dtype=bool)]

        r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
        r_ee = jnp.linalg.norm(ee, axis=-1, keepdims=True)
        return ae, ee, r_ae, r_ee

    def apply_env(self, out, r_ae):
        out = out * isotropic_envelope(r_ae,
                                       self.envelope_sigma, self.envelope_pi)

        out = out.reshape((sum(self.nelec), sum(self.nelec), self.ndets, 2, 2))
        return out[..., 0] + 1.j*out[..., 1]

    def construct_h_one_two(self, ae, ee, r_ae, r_ee):
        if r_ae is not None and ae is not None:
            h_one_in = jnp.concatenate(
                (ae.reshape(sum(self.nelec), -1), r_ae.reshape(sum(self.nelec), -1)), axis=-1)
        elif ae is not None:
            h_one_in = ae.reshape(sum(self.nelec), -1)
        else:
            h_one_in = None

        return h_one_in

    def construct_input(self, h_one):

        g_one = jnp.mean(h_one, axis=0, keepdims=True)
        f = jnp.concatenate(
            (h_one, jnp.tile(g_one, (sum(self.nelec), 1))), axis=-1)
        return f

    def __call__(self, positions: jnp.array, node_feats: jnp.array) -> jnp.ndarray:
        ae, ee, r_ae, r_ee = self.preprocess(positions)

        h_one_in = self.construct_h_one_two(ae, ee, r_ae, r_ee)
        f = self.construct_input(h_one_in)

        h_one_next = jnp.tanh(self.fc1_s(f))
        h_one_in = h_one_next

        f = self.construct_input(h_one_in)

        h_one_next = jnp.tanh(self.fc2_s(f))
        h_one_in = h_one_next

        f = self.construct_input(h_one_in)
        orbitals = self.fcf(f)

        return self.apply_env(orbitals, r_ae)


class ferminet(hk.Module):

    def __init__(self, simulation_cell, lattice, klist=None,  hidden_size=32,
                 hidden_size_single=256, ndets=1,  name=None):
        super().__init__(name=name)

        self.ndets = ndets
        self.init_output_size(simulation_cell)

        self.init_atom_params(simulation_cell)

        self.lattice = lattice

        self.hidden_size = hidden_size
        self.hidden_size_single = hidden_size_single

        self.fc1_d = hk.Linear(self.hidden_size)
        self.fc2_d = hk.Linear(self.hidden_size)
        self.fc3_d = hk.Linear(self.hidden_size)

        self.fcf = hk.Linear(self.output_size)

        self.fc1_s = hk.Linear(self.hidden_size_single)
        self.fc2_s = hk.Linear(self.hidden_size_single)
        self.fc3_s = hk.Linear(self.hidden_size_single)

        edge_index = np.stack((np.meshgrid(np.arange(sum(self.nelec)), np.arange(
            sum(self.nelec)))), -1)[~np.eye(sum(self.nelec), dtype=bool)]
        self.receivers = edge_index[:, 0]
        self.senders = edge_index[:, 1]

        # assumes nelec spin init.
        spinvecs = jnp.concatenate((jnp.tile(jnp.array([[0., 0., 1.]]), (self.nelec[0], 1)), jnp.tile(
            jnp.array([[0., 0., -1.]]), (self.nelec[1], 1))))
        # # maskup makes 1 for up elec sender, 0 for down elec sender. vice versa for maskdown.
        self.maskup = jnp.where(
            spinvecs[self.senders][:, -1] == 1, 1, 0)[:, None]
        self.maskdown = jnp.where(
            spinvecs[self.senders][:, -1] == -1, 1, 0)[:, None]

        self.hmaskup = jnp.where(spinvecs[:, -1] == 1, 1, 0)[:, None]
        self.hmaskdown = jnp.where(spinvecs[:, -1] == -1, 1, 0)[:, None]

        self.numberup = self.nelec[0]
        self.numberdown = self.nelec[1]

        self.klist = klist
        self.init_env_params()

    def init_output_size(self, simulation_cell):
        self.nelec = simulation_cell.nelec
        self.output_size = 2*2*sum(self.nelec)*self.ndets

    def init_atom_params(self, simulation_cell):
        self.atoms = simulation_cell.atom_coords()
        self.natoms = self.atoms.shape[0]

    def init_env_params(self):
        self.envelope_sigma = hk.get_parameter("env_sigma", shape=[
                                               1, self.natoms, self.output_size], dtype=jnp.float64, init=jnp.ones)
        self.envelope_pi = hk.get_parameter("env_pi", shape=[
                                            1, self.natoms, self.output_size], dtype=jnp.float64, init=jnp.ones)

    def preprocess(self, positions: jnp.array):
        ae, ee = get_rvec(positions, self.atoms)
        # removes the self-distance (otherwise need to deal with r=0)
        ee = ee[~np.eye(sum(self.nelec), dtype=bool)]

        r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
        r_ee = jnp.linalg.norm(ee, axis=-1, keepdims=True)
        return ae, ee, r_ae, r_ee

    def apply_env(self, out, r_ae):
        out = out * isotropic_envelope(r_ae,
                                       self.envelope_sigma, self.envelope_pi)
        out = out.reshape((sum(self.nelec), sum(self.nelec), self.ndets, 2, 2))
        return out[..., 0] + 1.j*out[..., 1]

    def construct_h_one_two(self, ae, ee, r_ae, r_ee):
        if r_ae is not None and ae is not None:
            h_one_in = jnp.concatenate(
                (ae.reshape(sum(self.nelec), -1), r_ae.reshape(sum(self.nelec), -1)), axis=-1)
        elif ae is not None:
            h_one_in = ae.reshape(sum(self.nelec), -1)
        else:
            h_one_in = None

        h_two_in = jnp.concatenate((ee, r_ee), axis=-1)
        return h_one_in, h_two_in

    def construct_input(self, h_one, h_two):

        g_two_up = scatter.scatter_sum(
            h_two*self.maskup, dst=self.receivers,  output_size=sum(self.nelec))/jnp.maximum(self.numberup, 1)
        g_two_down = scatter.scatter_sum(
            h_two*self.maskdown, dst=self.receivers,  output_size=sum(self.nelec))/jnp.maximum(self.numberdown, 1)

        # tile g_one
        if h_one is not None:
            g_one_up = jnp.sum(h_one * self.hmaskup, axis=0,
                               keepdims=True)/jnp.maximum(self.numberup, 1)
            g_one_down = jnp.sum(h_one * self.hmaskdown, axis=0,
                                 keepdims=True)/jnp.maximum(self.numberdown, 1)
            f = jnp.concatenate((h_one,  jnp.tile(g_one_up, (sum(self.nelec), 1)), jnp.tile(
                g_one_down, (sum(self.nelec), 1)), g_two_up, g_two_down), axis=-1)
        else:
            f = jnp.concatenate((g_two_up, g_two_down), axis=-1)
        return f

    def __call__(self, positions: jnp.array, node_feats: jnp.array) -> jnp.ndarray:
        ae, ee, r_ae, r_ee = self.preprocess(positions)

        h_one_in, h_two_in = self.construct_h_one_two(ae, ee, r_ae, r_ee)
        f = self.construct_input(h_one_in, h_two_in)

        h_one_next = jnp.tanh(self.fc1_s(f))
        h_two_next = jnp.tanh(self.fc1_d(h_two_in))

        h_one_in = h_one_next
        h_two_in = h_two_next

        f = self.construct_input(h_one_in, h_two_in)

        h_one_next = jnp.tanh(self.fc2_s(f))
        h_two_next = jnp.tanh(self.fc2_d(h_two_in))

        h_one_in = h_one_next
        h_two_in = h_two_next

        f = self.construct_input(h_one_in, h_two_in)

        h_one_next = jnp.tanh(self.fc3_s(f))
        h_two_next = jnp.tanh(self.fc3_d(h_two_in))

        h_one_in = h_one_next
        h_two_in = h_two_next

        f = self.construct_input(h_one_in, h_two_in)

        orbitals = self.fcf(f)

        return self.apply_env(orbitals, r_ae)


class spinornet(hk.Module):

    def __init__(self, simulation_cell, lattice, klist=None,  hidden_size=32,
                 hidden_size_single=256, ndets=1,  name=None):
        super().__init__(name=name)

        self.ndets = ndets
        self.init_output_size(simulation_cell)

        self.init_atom_params(simulation_cell)

        self.lattice = lattice

        self.hidden_size = hidden_size
        self.hidden_size_single = hidden_size_single

        self.fc1_d = hk.Linear(self.hidden_size)
        self.fc2_d = hk.Linear(self.hidden_size)

        self.fcf = hk.Linear(self.output_size)

        self.fc1_s = hk.Linear(self.hidden_size_single)
        self.fc2_s = hk.Linear(self.hidden_size_single)

        edge_index = np.stack((np.meshgrid(np.arange(sum(self.nelec)), np.arange(
            sum(self.nelec)))), -1)[~np.eye(sum(self.nelec), dtype=bool)]
        self.receivers = edge_index[:, 1]
        self.senders = edge_index[:, 0]

        self.klist = klist
        self.init_env_params()

    def init_output_size(self, simulation_cell):
        self.nelec = simulation_cell.nelec
        self.output_size = 2*2*sum(self.nelec)*self.ndets

    def init_atom_params(self, simulation_cell):
        self.atoms = simulation_cell.atom_coords()
        self.natoms = self.atoms.shape[0]

    def init_env_params(self):
        self.envelope_sigma = hk.get_parameter("env_sigma", shape=[
                                               1, self.natoms, self.output_size], dtype=jnp.float64, init=jnp.ones)
        self.envelope_pi = hk.get_parameter("env_pi", shape=[
                                            1, self.natoms, self.output_size], dtype=jnp.float64, init=jnp.ones)

    def preprocess(self, positions: jnp.array):
        ae, ee = get_rvec(positions, self.atoms)
        # removes the self-distance (otherwise need to deal with r=0)
        ee = ee[~np.eye(sum(self.nelec), dtype=bool)]

        r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
        r_ee = jnp.linalg.norm(ee, axis=-1, keepdims=True)
        return ae, ee, r_ae, r_ee

    def apply_env(self, out, r_ae):
        out = out * isotropic_envelope(r_ae,
                                       self.envelope_sigma, self.envelope_pi)
        out = out.reshape((sum(self.nelec), sum(self.nelec), self.ndets, 2, 2))
        return out[..., 0] + 1.j*out[..., 1]

    def construct_h_one_two(self, ae, ee, r_ae, r_ee):
        if r_ae is not None and ae is not None:
            h_one_in = jnp.concatenate(
                (ae.reshape(sum(self.nelec), -1), r_ae.reshape(sum(self.nelec), -1)), axis=-1)
        elif ae is not None:
            h_one_in = ae.reshape(sum(self.nelec), -1)
        else:
            h_one_in = None

        h_two_in = jnp.concatenate((ee, r_ee), axis=-1)
        return h_one_in, h_two_in

    def construct_input(self, h_one, h_two):

        g_two = scatter.scatter_sum(
            h_two, dst=self.receivers, output_size=sum(self.nelec))/(sum(self.nelec)-1)

        if h_one is not None:
            g_one = jnp.mean(h_one, axis=0, keepdims=True)
            f = jnp.concatenate(
                (h_one, jnp.tile(g_one, (sum(self.nelec), 1)), g_two), axis=-1)
        else:
            f = g_two
        return f

    def __call__(self, positions: jnp.array, node_feats: jnp.array) -> jnp.ndarray:
        ae, ee, r_ae, r_ee = self.preprocess(positions)

        h_one_in, h_two_in = self.construct_h_one_two(ae, ee, r_ae, r_ee)
        f = self.construct_input(h_one_in, h_two_in)

        h_one_next = jnp.tanh(self.fc1_s(f))
        h_two_next = jnp.tanh(self.fc1_d(h_two_in))

        h_one_in = h_one_next
        h_two_in = h_two_next

        f = self.construct_input(h_one_in, h_two_in)

        h_one_next = jnp.tanh(self.fc2_s(f))
        h_two_next = jnp.tanh(self.fc2_d(h_two_in))

        h_one_in = h_one_next
        h_two_in = h_two_next

        f = self.construct_input(h_one_in, h_two_in)

        orbitals = self.fcf(f)

        return self.apply_env(orbitals, r_ae)


def eval_phase(klist, positions, ndim=3):
    positions = positions.reshape([-1, ndim])
    kdotx = jnp.matmul(positions, klist.T)
    phases = jnp.exp(1j * kdotx)
    return phases


class _PBC(hk.Module):
    def __init__(self, simulation_cell, lattice, klist, ndets=1, name=None, feattype='ds', **kwargs):
        super().__init__(simulation_cell, lattice, klist=klist, ndets=ndets,  **kwargs)
        self.ns_tol = self.natoms // self.prim_natoms
        zmin = self.lattice[2, 2]/2
        plattice = simulation_cell.original_cell.a
        pzmin = plattice[2, 2]/2
        self.reciplatvec = jnp.linalg.inv(self.lattice)
        self.preciplatvec = jnp.linalg.inv(plattice)

        self.simulation_cell = simulation_cell
        self.pmetric = plattice.T @ plattice
        self.metric = lattice.T @ lattice
        self.feattype = feattype

    def init_output_size(self, simulation_cell):
        self.nelec = simulation_cell.nelec
        self.output_size = 2*2*sum(self.nelec)*self.ndets

    def init_atom_params(self, simulation_cell):
        self.atoms = simulation_cell.atom_coords()
        self.natoms = self.atoms.shape[0]
        self.prim_natoms = simulation_cell.original_cell.atom_coords().shape[0]

    def init_env_params(self):
        self.envelope_sigma = hk.get_parameter("env_sigma", shape=[
                                               1, 1, self.prim_natoms, self.output_size, 1], dtype=jnp.float64, init=jnp.ones)
        self.envelope_pi = hk.get_parameter("env_pi", shape=[
                                            1, 1, self.prim_natoms, self.output_size, 1], dtype=jnp.float64, init=jnp.ones)

    def preprocess(self, pos_pbc):
        pos_pbc, wrap = distance.enforce_pbc(
            self.lattice, self.reciplatvec, pos_pbc)
        if self.feattype == 'ds':
            aepbc, eepbc, r_ae, r_ee = construct_periodic_input_features(pos_pbc, self.atoms,
                                                                         simulation_cell=self.simulation_cell,
                                                                         )
            eepbc = eepbc[~np.eye(sum(self.nelec), dtype=bool)]
            r_ee = r_ee[~np.eye(sum(self.nelec), dtype=bool)]

        elif self.feattype == 'cas':
            ae, ee = get_rvec(pos_pbc, self.atoms)
            ee = ee[~np.eye(sum(self.nelec), dtype=bool)]
            if self.inc_ae:
                s_ae = jnp.einsum("...ij,jk->...ik", ae, self.preciplatvec)
                aepbc = jnp.concatenate(
                    (jnp.sin(2 * jnp.pi * s_ae), jnp.cos(2 * jnp.pi * s_ae)), axis=-1)
            # Two e features in phase coordinates
            s_ee = jnp.einsum("...ij,jk->...ik", ee, self.reciplatvec)

            eepbc = jnp.concatenate(
                (jnp.sin(2 * jnp.pi * s_ee), jnp.cos(2 * jnp.pi * s_ee)), axis=-1)
            if self.inc_rae:
                r_ae = periodic_norm(
                    self.pmetric, self.preciplatvec, s_ae)[..., None]

            r_ee = periodic_norm(
                self.metric, self.reciplatvec, s_ee)[..., None]

        if self.inc_ae and self.inc_rae:
            return aepbc, eepbc, r_ae, r_ee
        elif self.inc_ae:
            return aepbc, eepbc, None, r_ee
        else:
            return None, eepbc, None, r_ee

    def apply_env(self, out, r_ae):
        rae_split = jnp.split(r_ae, self.prim_natoms, axis=1)
        rae_split = jnp.concatenate([r[:, :, None] for r in rae_split], axis=2)
        env = isotropic_envelope(
            rae_split[:, :, :, None], self.envelope_sigma, self.envelope_pi)
        env = jnp.sum(env, axis=1)
        out = out * jnp.sum(env, axis=-1)
        out = out.reshape((sum(self.nelec), sum(self.nelec), self.ndets, 2, 2))
        return out[..., 0] + 1.j*out[..., 1]

    def __call__(self, positions: jnp.array, node_feats: jnp.array):

        out = super().__call__(positions, node_feats)
        phases = eval_phase(self.klist, positions, ndim=3)
        return out * phases[..., None, None]


class _pbcHEG(hk.Module):
    def __init__(self, simulation_cell, lattice, klist, hidden_size=None,  feattype='cas', inc_ae=False, inc_rae=False, name=None, **kwargs):
        super().__init__(simulation_cell, lattice, klist=klist,
                         hidden_size=hidden_size, feattype=feattype, name=name, **kwargs)
        self.inc_rae = inc_rae
        self.inc_ae = inc_ae

    def init_env_params(self):
        pass

    def apply_env(self, out, r_ae):

        out = out.reshape((sum(self.nelec), sum(self.nelec), self.ndets, 2, 2))
        return out[..., 0] + 1.j*out[..., 1]


class _PBC2d(hk.Module):
    def __init__(self, simulation_cell, lattice, klist, ndets=1, name=None, feattype='cas', **kwargs):
        super().__init__(simulation_cell, lattice, klist=klist, ndets=ndets,  **kwargs)
        self.ns_tol = self.natoms // self.prim_natoms
        plattice = simulation_cell.original_cell.a[:2, :2]
        self.invlat = jnp.linalg.inv(self.lattice)
        self.reciplatvec = jnp.linalg.inv(self.lattice[:2, :2])
        self.preciplatvec = jnp.linalg.inv(plattice)

        self.simulation_cell = simulation_cell
        self.pmetric = plattice.T @ plattice
        self.metric = lattice[:2, :2].T @ lattice[:2, :2]
        self.feattype = feattype

    def init_output_size(self, simulation_cell):
        self.nelec = simulation_cell.nelec
        self.output_size = 2*2*sum(self.nelec)*self.ndets

    def init_atom_params(self, simulation_cell):
        self.atoms = simulation_cell.atom_coords()[:, :2]
        self.natoms = self.atoms.shape[0]
        self.prim_natoms = simulation_cell.original_cell.atom_coords().shape[0]

    def init_env_params(self):
        self.envelope_sigma = hk.get_parameter("env_sigma", shape=[
                                               1, 1, self.prim_natoms, self.output_size, 1], dtype=jnp.float64, init=jnp.ones)
        self.envelope_pi = hk.get_parameter("env_pi", shape=[
                                            1, 1, self.prim_natoms, self.output_size, 1], dtype=jnp.float64, init=jnp.ones)

    def preprocess(self, pos_pbc):

        if self.feattype == 'cas':
            ae, ee = get_rvec(pos_pbc, self.atoms, ndim=2)
            ee = ee[~np.eye(sum(self.nelec), dtype=bool)]

            if self.inc_ae:
                s_ae = jnp.einsum("...ij,jk->...ik", ae, self.preciplatvec)
                aepbc = jnp.concatenate(
                    (jnp.sin(2 * jnp.pi * s_ae), jnp.cos(2 * jnp.pi * s_ae)), axis=-1)
            # Two e features in phase coordinates
            s_ee = jnp.einsum("...ij,jk->...ik", ee, self.reciplatvec)

            eepbc = jnp.concatenate(
                (jnp.sin(2 * jnp.pi * s_ee), jnp.cos(2 * jnp.pi * s_ee)), axis=-1)
            if self.inc_rae:
                r_ae = periodic_norm(
                    self.pmetric, self.preciplatvec, s_ae)[..., None]

            r_ee = periodic_norm(
                self.metric, self.reciplatvec, s_ee)[..., None]

        if self.inc_ae and self.inc_rae:
            return aepbc, eepbc, r_ae, r_ee
        elif self.inc_ae:
            return aepbc, eepbc, None, r_ee
        else:
            return None, eepbc, None, r_ee

    def apply_env(self, out, r_ae):
        rae_split = jnp.split(r_ae, self.prim_natoms, axis=1)
        rae_split = jnp.concatenate([r[:, :, None] for r in rae_split], axis=2)
        env = isotropic_envelope(
            rae_split[:, :, :, None], self.envelope_sigma, self.envelope_pi)
        env = jnp.sum(env, axis=1)
        out = out * jnp.sum(env, axis=-1)
        out = out.reshape((sum(self.nelec), sum(self.nelec), self.ndets, 2, 2))
        return out[..., 0] + 1.j*out[..., 1]

    def __call__(self, positions: jnp.array, node_feats: jnp.array):
        positions, wrap = distance.enforce_pbc(
            self.lattice, self.invlat, positions)
        positions = jnp.delete(positions, jnp.arange(
            2, positions.size, 3), assume_unique_indices=True)

        out = super().__call__(positions, node_feats)
        phases = eval_phase(self.klist[:, :2], positions, ndim=2)
        return out * phases[..., None]


class spinornet_pbc(_PBC, spinornet):
    def __init__(self, simulation_cell, lattice, klist,
                 hidden_size=32, hidden_size_single=256,  feattype='ds', ndets=1, name=None):
        super().__init__(simulation_cell, lattice, klist=klist, hidden_size=hidden_size,
                         hidden_size_single=hidden_size_single, feattype=feattype, ndets=ndets, name=name)
        self.inc_rae = True
        self.inc_ae = True


class spinornet_pbc2d(_PBC2d, spinornet_pbc):
    def __init__(self, simulation_cell, lattice, klist,  hidden_size=32, hidden_size_single=256,  feattype='cas', ndets=1, name=None):
        super().__init__(simulation_cell, lattice, klist=klist,  hidden_size=hidden_size,
                         hidden_size_single=hidden_size_single, feattype=feattype, ndets=ndets,  name=name)
        self.inc_rae = True
        self.inc_ae = True


class spinornet_pbc2dHEG(_pbcHEG, spinornet_pbc2d, spinornet):
    def __init__(self, simulation_cell, lattice, klist,
                 hidden_size=32, hidden_size_single=256,  feattype='cas',
                 inc_ae=False, inc_rae=False, ndets=1,  name=None):
        super().__init__(simulation_cell, lattice, klist=klist, hidden_size=hidden_size,
                         hidden_size_single=hidden_size_single,  feattype=feattype,
                         inc_ae=inc_ae, inc_rae=inc_rae, ndets=ndets,  name=name)

    def __call__(self, positions: jnp.array, node_feats: jnp.array):
        positions = jnp.delete(positions, jnp.arange(
            2, positions.size, 3), assume_unique_indices=True)
        out = spinornet.__call__(self, positions, node_feats)
        return out


def scaled_f(w):
    """
    see Phys. Rev. B 94, 035157
    :param w: projection of position vectors on reciprocal vectors.
    :return: function f in the ref.
    """
    return jnp.abs(w) * (1 - jnp.abs(w / jnp.pi) ** 3 / 4.)


def scaled_g(w):
    """
    see Phys. Rev. B 94, 035157
    :param w: projection of position vectors on reciprocal vectors.
    :return: function g in the ref.
    """
    return w * (1 - 3. / 2. * jnp.abs(w / jnp.pi) + 1. / 2. * jnp.abs(w / jnp.pi) ** 2)


def nu_distance(xea, a, b):
    """
    see Phys. Rev. B 94, 035157
    :param xea: relative distance between electrons and atoms
    :param a: lattice vectors of primitive cell divided by 2\pi.
    :param b: reciprocal vectors of primitive cell.
    :return: periodic generalized relative and absolute distance of xea.
    """
    w = jnp.einsum('...ijk,lk->...ijl', xea, b)
    mod = (w + jnp.pi) // (2 * jnp.pi)
    w = (w - mod * 2 * jnp.pi)
    r1 = (jnp.linalg.norm(a, axis=-1) * scaled_f(w)) ** 2
    sg = scaled_g(w)
    rel = jnp.einsum('...i,ij->...j', sg, a)
    r2 = jnp.einsum('ij,kj->ik', a, a) * (sg[..., :, None] * sg[..., None, :])
    result = jnp.sum(r1, axis=-1) + jnp.sum(r2 *
                                            (jnp.ones(r2.shape[-2:]) - jnp.eye(r2.shape[-1])), axis=[-1, -2])
    sd = result ** 0.5
    return sd, rel


def periodic_norm(metric, recip_vecs, r):
    a = (1 - jnp.cos(2 * jnp.pi * r))
    b = jnp.sin(2 * jnp.pi * r)
    cos_term = jnp.einsum('...m,mn,...n->...', a, metric, a)
    sin_term = jnp.einsum('...m,mn,...n->...', b, metric, b)
    return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)


def construct_periodic_input_features(
        x: jnp.ndarray,
        atoms: jnp.ndarray,
        simulation_cell=None,
        ndim: int = 3):
    """Constructs a periodic generalized inputs to Fermi Net from raw electron and atomic positions.
    see Phys. Rev. B 94, 035157
      Args:
        x: electron positions. Shape (nelectrons*ndim,).
        atoms: atom positions. Shape (natoms, ndim).
        simulation_cel: spinornet system.HEGCell object
        ndim: dimension of system. Change only with caution.
      Returns:
        ae, ee, r_ae, r_ee tuple, where:
          ae: atom-electron vector. Shape (nelectron, natom, 3).
          ee: atom-electron vector. Shape (nelectron, nelectron, 3).
          r_ae: atom-electron distance. Shape (nelectron, natom, 1).
          r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
        The diagonal terms in r_ee are masked out such that the gradients of these
        terms are also zero.
      """
    primitive_cell = simulation_cell.original_cell
    x = x.reshape(-1, ndim)
    n = x.shape[0]
    prim_x, _ = distance.enforce_pbc2(primitive_cell.a, x)

    prim_xea = prim_x[..., None, :] - atoms
    prim_periodic_sea, prim_periodic_xea = nu_distance(prim_xea,
                                                       primitive_cell.AV,
                                                       primitive_cell.BV)
    prim_periodic_sea = prim_periodic_sea[..., None]

    sim_x, _ = distance.enforce_pbc2(simulation_cell.a, x)
    sim_xee = sim_x[None, :, :] - sim_x[:, None, :]

    sim_periodic_see, sim_periodic_xee = nu_distance(sim_xee + jnp.eye(n)[..., None],
                                                     simulation_cell.AV,
                                                     simulation_cell.BV)
    sim_periodic_see = sim_periodic_see * (1.0 - jnp.eye(n))
    sim_periodic_see = sim_periodic_see[..., None]

    sim_periodic_xee = sim_periodic_xee * (1.0 - jnp.eye(n))[..., None]

    return prim_periodic_xea, sim_periodic_xee, prim_periodic_sea, sim_periodic_see


def init_nn_psi_params(orbital_method, factor_method, ndets, simulation_cell, key,
                       pos, ft):
    # some params can be inited w haiku

    params = orbital_method.init(key, pos)
    nnparams = factor_method.init(key, pos, ft)
    params = hk.data_structures.merge(params, nnparams)

    # if using some additional params
    # params['orbitals'] = {}
    return params


def logdet_matmul(xs: Sequence[jnp.ndarray],
                  w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Combines determinants and takes dot product with weights in log-domain.

    We use the log-sum-exp trick to reduce numerical instabilities.

    Args:
      xs: orbitals in each determinant. 
      w: weight of each determinant. If none, a uniform weight is assumed.

    Returns:
      sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
      determinant (or product of the i-th determinant in each spin channel, if
      full_det is not used).
    """
    sign_in, slogdet = jnp.linalg.slogdet(xs)
    slogdet_max = jnp.max(slogdet)
    det = sign_in * jnp.exp(slogdet-slogdet_max)
    result = jnp.sum(det)
    sign_out = jnp.exp(1j*jnp.angle(result))
    slog_out = jnp.log(jnp.abs(result)) + slogdet_max
    return sign_out, slog_out


def eval_func(orbital_method, nnfactor, params, x, ft,
              method_name='eval_slogdet'):
    '''
    generates the wavefunction of simulation cell.
    :param orbital_method: Orbital method.
    :param nnfactor: neural network method.
    :param params: parameter dict
    :param x: The input data, last dim is 3N with N being num electrons.
    :param ft: node_feats. could include spin info.
    :param method_name: specify the returned function of wavefunction
    :return: required wavefunction
    '''

    orbitals = orbital_method.apply(params, x)
    if method_name == 'eval_hf_orb':
        return orbitals

    nnf = nnfactor.apply(params, x, ft)

    nnorbitals = (nnf) * orbitals[:, :, None]

    spinor = ft

    spinorbitals = jnp.einsum("es,eods -> deo ", jnp.conj(spinor), nnorbitals)

    if method_name == 'eval_slogdet':
        _, result = logdet_matmul(spinorbitals)
        result = result
    elif method_name == 'eval_logdet':
        sign, slogdet = logdet_matmul(spinorbitals)
        result = jnp.log(sign) + slogdet
        result = result
    elif method_name == 'eval_mats':
        result = nnorbitals
    else:
        raise ValueError('Unrecognized method name')

    return result


def make_nn_psi(simulation_cell=None,
                lattice=None,
                hartree_fock=None,
                klist=None,
                ndets=None,
                method_name='eval_slogdet',
                network_method=None,
                hf_method=None,
                alpha=None,
                kwargs={},
                ):
    '''
    generates the wavefunction of simulation cell.
    :param simulation_cell: simulation cell
    :param lattice: required if periodic cell.
    :param hartree_fock: spinornet.hf.SCFMol object, required if using HFOrbital
    :param klist: kpoints.
    :param ndets: number of determinants.
    :param method_name: specify the returned function
    :param network_method: type of neural network to use.
    :param hf_method: type of HF orbital to use.
    :param alpha: parameter for Rashba strength.
    :return: a haiku like module which contain init and apply method. init is used to initialize the parameter of
    network and apply method perform the calculation.
    '''

    if method_name not in ['eval_slogdet', 'eval_mats', 'eval_logdet', 'eval_hf_orb']:
        raise ValueError('Method name is not in class dir.')
    method = namedtuple('method', ['init', 'apply'])
    if network_method not in ['ferminet', 'spinornet', 'spinornet_single', 'spinornet_pbc', 'spinornet_pbc2dHEG']:
        raise ValueError('Network method name is not in class dir.')
    networks = {
        'ferminet': ferminet, 'spinornet': spinornet, 'spinornet_single': spinornet_single, 'spinornet_pbc': spinornet_pbc,  'spinornet_pbc2dHEG': spinornet_pbc2dHEG,
    }
    if hf_method not in ['HFOrbital', 'HFOrbitalcol', 'HFOrbitalHEG2d', 'HFOrbitalHEG2dcol', 'HFOrbitalHEG2dRb']:
        raise ValueError('Hf method name is not in class dir.')
    hfmethod = {'HFOrbital': HFOrbital, 'HFOrbitalcol': HFOrbitalcol,
                'HFOrbitalHEG2d': HFOrbitalHEG2d, 'HFOrbitalHEG2dcol': HFOrbitalHEG2dcol, 'HFOrbitalHEG2dRb': HFOrbitalHEG2dRb}

    @hk.without_apply_rng
    @hk.transform
    def nnfactor(x, ft):
        net = networks[network_method](simulation_cell=simulation_cell,
                                       lattice=lattice,
                                       klist=klist,
                                       ndets=ndets,
                                       **kwargs)
        return net(x, ft)

    if hf_method in ['HFOrbitalHEG2dRb']:
        @hk.without_apply_rng
        @hk.transform
        def hforbitals(pos):
            return hfmethod[hf_method](hartree_fock, lb=alpha)(pos)

    elif hf_method in ['HFOrbital', 'HFOrbitalcol']:
        hforb_init = hfmethod[hf_method].from_pyscf(hartree_fock)

        @hk.without_apply_rng
        @hk.transform
        def hforbitals(pos):
            return hfmethod[hf_method](*hforb_init)(pos)

    else:
        @hk.without_apply_rng
        @hk.transform
        def hforbitals(pos):
            return hfmethod[hf_method](hartree_fock)(pos)

    init = functools.partial(init_nn_psi_params,
                             hforbitals, nnfactor, ndets, simulation_cell)

    pred_out = functools.partial(eval_func,
                                 hforbitals,
                                 nnfactor,
                                 method_name=method_name)

    method.init = init
    method.apply = pred_out
    return method
