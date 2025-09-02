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

from spinornet import network
import jax.numpy as jnp
from typing import Union
import numpy as np
import jax
from jax import lax
from spinornet import ewaldsum
from spinornet import ewald2dsum
from spinornet import spins_jax
import folx

Array = Union[jnp.ndarray, np.ndarray]


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
    """Returns the electron-electron potential.

    Args:
      r_ee: Shape (nelectrons, nelectrons, :). r_ee[i,j,0] gives the distance
        between electrons i and j. Other elements in the final axes are not
        required.
    """
    return jnp.sum(jnp.triu(1 / r_ee, k=1))


def potential_electron_nuclear(charges, r_ae: Array) -> jnp.ndarray:
    """Returns the electron-nuclear potential.

    Args:
      charges: Shape (natoms). Nuclear charges of the atoms.
      r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
        electron i and atom j.
    """
    return -jnp.sum(charges / r_ae)


def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
    """Returns the electron-nuclearpotential.

    Args:
      charges: Shape (natoms). Nuclear charges of the atoms.
      atoms: Shape (natoms, ndim). Positions of the atoms.
    """
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    return jnp.sum(
        jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def local_kinetic_energy(f, lap_method='scan'):
    '''
    :param f: function return the logdet of wavefunction
    :return: local kinetic energy
    '''
    if lap_method == 'folx':
        def _lapl_over_f(params, x, ft):
            def f_closure(x): return f(params, x, ft)
            f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
            output = f_wrapped(x)
            grad2 = jnp.sum(output.jacobian.dense_array ** 2)
            result = - (output.laplacian + grad2) / 2
            return result

    else:
        def _lapl_over_f(params, x, ft):
            ne = x.shape[-1]
            eye = jnp.eye(ne)
            grad_f = jax.grad(f, argnums=1)
            def grad_f_closure(y): return grad_f(params, y, ft)

            # vmap version
            if lap_method == 'vmap':
                primal, tangent = jax.vmap(jax.jvp, (None, None, (0,)))(
                    grad_f_closure, (x,), (eye,))
                grad2 = jnp.sum(primal[0]**2)
                return -0.5 * (jnp.trace(tangent) + grad2)

            # scan version. (scan is better than fori_loop.)
            else:
                primal, dgrad_f = jax.linearize(grad_f_closure, x)
                def hessian_diagonal(i): return dgrad_f(eye[i])[i]
                _, diagonal = lax.scan(
                    lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=ne)
                grad2 = jnp.sum(primal**2)
                result = -0.5 * (jnp.sum(diagonal) + grad2)
                return result

    return _lapl_over_f


def local_kinetic_energy_complex(f, lap_method='scan'):
    '''
    :param f: function return the logdet of wavefunction
    :return: local kinetic energy
    '''
    vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

    if lap_method == 'folx':
        def _lapl_over_f(params, x, ft):
            def f_closure_real(x): return f(params, x, ft).real
            f_wrapped_real = folx.forward_laplacian(
                f_closure_real, sparsity_threshold=6)

            def f_closure_imag(x): return f(params, x, ft).imag
            f_wrapped_imag = folx.forward_laplacian(
                f_closure_imag, sparsity_threshold=6)
            output_real = f_wrapped_real(x)
            output_imag = f_wrapped_imag(x)
            grad2re = jnp.sum(output_real.jacobian.dense_array **
                              2 - output_imag.jacobian.dense_array**2)
            realp = output_real.laplacian + grad2re
            grad2im = jnp.sum(2*output_real.jacobian.dense_array *
                              output_imag.jacobian.dense_array)
            imagp = output_imag.laplacian + grad2im
            return -0.5 * (realp + imagp * 1.j)
    else:
        def _lapl_over_f(params, x, ft):
            ne = x.shape[-1]
            eye = jnp.eye(ne)
            grad_f_real = jax.grad(
                lambda p, y, ft: f(p, y, ft).real, argnums=1)
            grad_f_imag = jax.grad(
                lambda p, y, ft: f(p, y, ft).imag, argnums=1)

            def grad_f_closure_real(y): return grad_f_real(params, y, ft)
            def grad_f_closure_imag(y): return grad_f_imag(params, y, ft)

            # vmap version
            if lap_method == 'vmap':
                primal_real, tangent_real = vjvp(
                    grad_f_closure_real, (x,), (eye,))
                primal_imag, tangent_imag = vjvp(
                    grad_f_closure_imag, (x,), (eye,))
                grad2re = jnp.sum(primal_real[0]**2 - primal_imag[0]**2)
                realp = jnp.trace(tangent_real) + grad2re
                grad2im = jnp.sum(2*primal_real[0]*primal_imag[0])
                imagp = jnp.trace(tangent_imag) + grad2im
                return -0.5 * (realp + imagp * 1.j)

            # scan version.
            else:
                primal_real, dgrad_f_real = jax.linearize(
                    grad_f_closure_real, x)
                primal_imag, dgrad_f_imag = jax.linearize(
                    grad_f_closure_imag, x)

                def hessian_diagonal_real(i): return dgrad_f_real(eye[i])[i]
                def hessian_diagonal_imag(i): return dgrad_f_imag(eye[i])[i]
                _, diagonal_real = lax.scan(
                    lambda i, _: (i + 1, hessian_diagonal_real(i)), 0, None, length=ne)
                _, diagonal_imag = lax.scan(
                    lambda i, _: (i + 1, hessian_diagonal_imag(i)), 0, None, length=ne)
                grad2re = jnp.sum(primal_real**2 - primal_imag**2)
                grad2im = jnp.sum(2*primal_real*primal_imag)
                realp = jnp.sum(diagonal_real) + grad2re
                imagp = jnp.sum(diagonal_imag) + grad2im
                result = -0.5 * (realp + imagp * 1.j)
                return result

    return _lapl_over_f


def local_rashba_soc(f, alpha, use_vmap=False):
    vmapf = jax.vmap(f, (None, None, 0))

    def _rashba_over_f(params, x, ft):
        nelec = ft.shape[0]
        ftx = spins_jax.flip_spinx(ft)
        fty = spins_jax.flip_spiny(ft)
        grad_f_real = jax.grad(lambda p, y, ft: f(p, y, ft).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y, ft: f(p, y, ft).imag, argnums=1)
        def grad_f_closure_real(y, ft): return grad_f_real(params, y, ft)
        def grad_f_closure_imag(y, ft): return grad_f_imag(params, y, ft)

        ftx1 = jnp.tile(ft[None, ...], (nelec, 1, 1)
                        ).at[np.eye(nelec, dtype=bool)].set(ftx)
        fty1 = jnp.tile(ft[None, ...], (nelec, 1, 1)
                        ).at[np.eye(nelec, dtype=bool)].set(fty)

        orig = f(params, x, ft)
        newx = vmapf(params, x, ftx1)
        newy = vmapf(params, x, fty1)

        weightx = jnp.exp(newx - orig)
        weighty = jnp.exp(newy - orig)

        if use_vmap:
            vgrad_f_closure_real = jax.vmap(grad_f_closure_real, (None, 0))
            vgrad_f_closure_imag = jax.vmap(grad_f_closure_imag, (None, 0))
            pysx_real = vgrad_f_closure_real(
                x, ftx1)[np.arange(nelec), np.arange(1, 3*nelec, 3)]
            pysx_imag = vgrad_f_closure_imag(
                x, ftx1)[np.arange(nelec), np.arange(1, 3*nelec, 3)]

            pxsy_real = vgrad_f_closure_real(
                x, fty1)[np.arange(nelec), np.arange(0, 3*nelec, 3)]
            pxsy_imag = vgrad_f_closure_imag(
                x, fty1)[np.arange(nelec), np.arange(0, 3*nelec, 3)]

            res = jnp.sum(pxsy_real*weighty - pysx_real*weightx) + \
                jnp.sum(pxsy_imag*weighty - pysx_imag*weightx) * 1.j

            return 1.j * alpha * res
        else:
            def pysx_real(i): return grad_f_closure_real(x, ftx1[i])[i*3+1]
            def pysx_imag(i): return grad_f_closure_imag(x, ftx1[i])[i*3+1]
            def pxsy_real(i): return grad_f_closure_real(x, fty1[i])[i*3]
            def pxsy_imag(i): return grad_f_closure_imag(x, fty1[i])[i*3]
            _, pysx_real_sum = lax.scan(
                lambda i, _: (i + 1, pysx_real(i)), 0, None, length=nelec)
            _, pysx_imag_sum = lax.scan(
                lambda i, _: (i + 1, pysx_imag(i)), 0, None, length=nelec)
            _, pxsy_real_sum = lax.scan(
                lambda i, _: (i + 1, pxsy_real(i)), 0, None, length=nelec)
            _, pxsy_imag_sum = lax.scan(
                lambda i, _: (i + 1, pxsy_imag(i)), 0, None, length=nelec)
            res = jnp.sum(pxsy_real_sum*weighty - pysx_real_sum*weightx) + \
                jnp.sum(pxsy_imag_sum*weighty - pysx_imag_sum*weightx)*1.j
            return 1.j * alpha * res
    return _rashba_over_f


def make_local_energy(f, mol, lap_method='scan'):
    """Returns a callable e_l(params, pos, feats) which evaluates the local
     energy of the wavefunction given the parameters params, and a 
     single MCMC configuration in data (with electron pos and electron feats).
    """
    lap_over_f = local_kinetic_energy(f, lap_method)
    ii = potential_nuclear_nuclear(mol.atom_charges(), mol.atom_coords())

    def _e_l(params, pos, feats) -> jnp.ndarray:
        """Returns the total energy.

        Args:
          params: network parameters.
          pos: position.
          feats: node features.
        """
        ae, ee = network.get_rvec(pos, mol.atom_coords())
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ee = jnp.linalg.norm(ee, axis=-1)

        eep = potential_electron_electron(r_ee)
        ei = potential_electron_nuclear(mol.atom_charges(), r_ae)
        ke = lap_over_f(params, pos, feats)
        return ke, eep + ei + ii
    return _e_l


def local_ewald_energy(simulation_cell, heg=False):
    """
    generate local energy of ewald part.
    :param simulation_cell:
    :return:
    """
    ewald = ewaldsum.EwaldSum(simulation_cell, heg=heg)

    def _local_ewald_energy(x):
        energy = ewald.energy(x)
        return sum(energy)

    return _local_ewald_energy


def local_ewald2d_energy(simulation_cell, heg=False):
    ewald = ewald2dsum.Ewaldjax(simulation_cell, heg=heg)

    def _local_ewald_energy(x):
        energy = ewald.energy(x)
        return sum(energy)
    return _local_ewald_energy


def make_ewald_energy(f, mol, heg=False, lap_method='scan'):
    """Returns a callable e_l(params, pos, feats) which evaluates the local
      energy of the wavefunction given the parameters params, and a 
      single MCMC configuration in data (with electron pos and electron feats).
      for pbc. 
    """
    ew = local_ewald_energy(mol, heg=heg)
    ke = local_kinetic_energy_complex(f, lap_method)

    def _local_energy(params, x, ft):
        kinetic = ke(params, x, ft)
        ewald = ew(x)
        return kinetic, ewald

    return _local_energy


def make_ewald2d_energy(f, mol, alpha=1, heg=False, lap_method='scan', use_vmap=False):
    """Returns a callable e_l(params, pos, feats) which evaluates the local
      energy of the wavefunction given the parameters params, and a 
      single MCMC configuration in data (with electron pos and electron feats).
      for pbc. 
    """
    ew = local_ewald2d_energy(mol, heg=heg)
    ke = local_kinetic_energy_complex(f, lap_method)
    if alpha != 0:
        rsoc = local_rashba_soc(f, alpha, use_vmap)

    def _local_energy(params, x, ft):
        kinetic = ke(params, x, ft)
        ewald = ew(x)
        if alpha != 0:
            rashba = rsoc(params, x, ft)
        else:
            rashba = 0
        return kinetic, ewald, rashba

    return _local_energy


def make_energy_complex_mol(f, mol, heg=False, lap_method='scan'):
    lap_over_f = local_kinetic_energy_complex(f, lap_method)
    ii = potential_nuclear_nuclear(mol.atom_charges(), mol.atom_coords())

    def _e_l(params, pos, feats) -> jnp.ndarray:
        """Returns the total energy.

        Args:
          params: network parameters.
          pos: position.
          feats: node features.
        """
        ae, ee = network.get_rvec(pos, mol.atom_coords())
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ee = jnp.linalg.norm(ee, axis=-1)

        eep = potential_electron_electron(r_ee)
        ei = potential_electron_nuclear(mol.atom_charges(), r_ae)
        ke = lap_over_f(params, pos, feats)
        return ke, eep + ei + ii
    return _e_l
