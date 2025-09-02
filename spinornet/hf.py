# MIT License
#
# Copyright (c) 2019 Lucas K Wagner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# This file may have been modified by Spinornet authors.
# All Spinornet modifications are Copyright 2025 Spinornet authors.

from pyscf.pbc import gto, scf
from spinornet import distance
from spinornet import supercell
import numpy as np
from pyscf.gto import Mole
from pyscf.scf import RHF
import pyqmc.api as pyq
import pyqmc.gpu as gpu
import jax
from pyscf.scf.addons import remove_linear_dep_
from spinornet import init_guess
import jax.numpy as jnp


class SCFMol:
    def __init__(self, mol: Mole, wf=None):
        self._mol = mol
        self.wf = wf
        self.nelec = mol.nelec

    def run(self):
        if self.wf is None:
            self.mf = RHF(self._mol.build())
            self.mf.kernel()
            self.wf, _ = pyq.generate_slater(self._mol, self.mf)

    def aos(self, eval_str, coords):
        mycoords = coords
        mycoords = mycoords.reshape((-1, 3))
        aos = gpu.cp.asarray(
            [self.wf.orbitals._mol.eval_gto(eval_str, mycoords)])
        if len(aos.shape) == 4:  # if derivatives are included
            return aos.reshape((1, aos.shape[1], *mycoords.shape[:-1], aos.shape[-1]))
        else:
            return aos.reshape((1, *mycoords.shape[:-1], aos.shape[-1]))

    def eval_orb_mat(self, coords):
        aos = self._mol.eval_gto("GTOval_sph", coords.reshape((-1, 3)))
        orbitals = self.mf.mo_coeff[:, :max(self.nelec)]
        mos = aos.dot(orbitals).reshape(coords.shape[0], sum(self.nelec), -1)

        spin_mos = np.zeros((coords.shape[0], sum(
            self.nelec), sum(self.nelec), 2), dtype=np.complex128)
        spin_mos[:, :, 0:self.nelec[0], 0] = mos[:, :, :self.nelec[0]]
        spin_mos[:, :, self.nelec[0]:, 1] = mos[:, :, :self.nelec[1]]
        return spin_mos

    def generate_wf_samples_jax(self, seed, nconfig, init_width=0.5):
        cell = init_guess.pyscf_to_cell(cell=self._mol)
        configs = init_guess.init_electrons_mol_jax(
            seed, cell, self.nelec, nconfig, init_width=init_width)
        return configs


class SCF:
    def __init__(self, cell, twist=np.ones(3)*0.5):
        """
        Hartree Fock wave function class for QMC simulation

        :param cell: pyscf.pbc.gto.Cell, simulation object
        :param twist:np.array with shape [3]
        """
        self.coeff_key = ("mo_coeff_alpha", "mo_coeff_beta")
        self.param_split = {}
        self.parameters = {}
        self.k_split = {}
        self.ns_tol = cell.scale
        self.simulation_cell = cell
        self.primitive_cell = cell.original_cell
        self.nelec = self.simulation_cell.nelec
        self.kpts = supercell.get_supercell_kpts(self.simulation_cell)
        self.kpts = self.kpts + \
            np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
        if hasattr(self.simulation_cell, 'hf_type'):
            hf_type = self.simulation_cell.hf_type
        else:
            hf_type = 'rhf'

        if hf_type == 'uhf':
            self.kmf = scf.KUHF(self.primitive_cell,
                                exxdiv='ewald', kpts=self.kpts).density_fit()

            # break initial guess symmetry for UHF
            dm_up, dm_down = self.kmf.get_init_guess(key='minao')
            dm_down[:, :2, :2] = 0
            dm = (dm_up, dm_down)
        elif hf_type == 'rhf':
            self.kmf = scf.KHF(self.primitive_cell,
                               exxdiv='ewald', kpts=self.kpts).density_fit()
            dm = self.kmf.get_init_guess()
        else:
            raise ValueError('Unrecognized Hartree Fock type.')
        self.kmf.max_cycle = 0
        self.hf_energy = self.kmf.kernel(dm)

    def init_scf(self):
        """
        initialization function to set up HF ansatz.
        """
        self.klist = []
        for s, key in enumerate(self.coeff_key):
            mclist = []
            for k in range(self.kmf.kpts.shape[0]):
                # restrict or not
                if len(self.kmf.mo_coeff[0][0].shape) == 2:
                    mca = self.kmf.mo_coeff[s][k][:, np.asarray(
                        self.kmf.mo_occ[s][k] > 0.9)]
                else:
                    minocc = (0.9, 1.1)[s]
                    mca = self.kmf.mo_coeff[k][:, np.asarray(
                        self.kmf.mo_occ[k] > minocc)]
                mclist.append(mca)
            self.param_split[key] = np.cumsum([m.shape[1] for m in mclist])
            self.parameters[key] = np.concatenate(mclist, axis=-1)
            self.k_split[key] = np.array([m.shape[1] for m in mclist])
            self.klist.append(np.concatenate([np.tile(kpt[None, :], (split, 1))
                                              for kpt, split in
                                              zip(self.kmf.kpts, self.k_split[self.coeff_key[s]])]))
        self.klist = np.concatenate(self.klist)

    def eval_mos_pbc(self, aos, s):
        """
        eval the molecular orbital values.
        :param aos: atomic orbital values.
        :param s: spin index.
        :return: molecular orbital values.
        """
        c = self.coeff_key[s]
        p = np.split(self.parameters[c], self.param_split[c], axis=-1)
        mo = [ao.dot(p[k]) for k, ao in enumerate(aos)]
        return np.concatenate(mo, axis=-1)

    def eval_orbitals_pbc(self, coord, eval_str="GTOval_sph"):
        """
        eval the atomic orbital valus of HF.
        :param coord: electron walkers with shape [batch, ne * ndim].
        :param eval_str:
        :return: atomic orbital valus of HF.
        """
        prim_coord, wrap = distance.np_enforce_pbc(
            self.primitive_cell.a, coord.reshape([coord.shape[0], -1]))
        prim_coord = prim_coord.reshape([-1, 3])
        wrap = wrap.reshape([-1, 3])
        ao = self.primitive_cell.eval_gto(
            "PBC" + eval_str, prim_coord, kpts=self.kmf.kpts)

        kdotR = np.einsum('ij,kj,nk->in', self.kmf.kpts,
                          self.primitive_cell.a, wrap)
        wrap_phase = np.exp(1j*kdotR)
        ao = [ao[k] * wrap_phase[k][:, None]
              for k in range(len(self.kmf.kpts))]

        return ao

    def eval_orb_mat(self, coord):
        """
        eval the orbital matrix of HF.
        :param coord: electron walkers with shape [batch, ne * ndim].
        :return: orbital matrix of HF.
        """
        batch, nelecndim = coord.shape
        nelec = nelecndim // 3
        aos = self.eval_orbitals_pbc(coord)
        aos_shape = (self.ns_tol, batch, nelec, -1)

        aos = np.reshape(aos, aos_shape)

        spin_mos = np.zeros(
            (batch, sum(self.nelec), sum(self.nelec)), dtype=np.complex128)
        for s in [0, 1]:
            i0, i1 = s * self.nelec[0], self.nelec[0] + s * self.nelec[1]
            mo = self.eval_mos_pbc(aos, s).reshape([batch, nelec, -1])
            spin_mos[:, i0:i1, i0:i1] = mo[:, i0:i1]
        return spin_mos

    def generate_wf_samples_jax(self, seed, nconfig, init_width=0.5):
        cell = init_guess.pyscf_to_cell(cell=self.simulation_cell)
        configs = init_guess.init_electrons(
            seed, cell, self.nelec, nconfig, init_width=init_width, latvec=self.simulation_cell.a)
        return configs.reshape((nconfig, -1))


class HEGscf2d:
    def __init__(self, cell, twist=np.ones(3)*0.5):
        self._mol = cell
        self.nelec = cell.nelec
        self.lattice = cell.a
        self.klist = cell.klist

    def generate_wf_samples_jax(self, seed, nconfig, init_width=0.5):
        configs = jax.random.uniform(seed, shape=(nconfig, sum(self.nelec), 3))
        configs = configs @ self.lattice
        configs = configs.at[:, :, 2].set(0)
        return configs.reshape((nconfig, -1)).astype(np.float64)

    def eval_orb_mat(self, coord):
        batch, nelecndim = coord.shape
        datatest = coord.reshape(-1, sum(self.nelec), 3)
        kdotx = np.matmul(datatest[:, :, :2], self.klist[:, :2].T)
        # this assumes that the non-zero electron spins are on nelec[0].
        # also assumes even # of total electrons if half-up, half-down.
        waves = np.cos(kdotx[:, :, :sum(self.nelec)]) + \
            1.j*np.sin(kdotx[:, :, :sum(self.nelec)])

        spin_mos = np.zeros((batch, sum(self.nelec), sum(
            self.nelec), 2), dtype=np.complex128)
        spin_mos[:, :, 0:self.nelec[0], 0] = waves[:, :, :self.nelec[0]]
        spin_mos[:, :, self.nelec[0]:, 1] = waves[:, :, :self.nelec[1]]
        return spin_mos
