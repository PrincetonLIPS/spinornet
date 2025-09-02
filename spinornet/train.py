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

from spinornet import hamiltonian
import jax.numpy as jnp
import jax
import chex
import spinornet.kfac_alpha as kfac_alpha
import functools
from typing import Optional
from spinornet import init_guess
from spinornet import constants
import folx


@chex.dataclass
class AuxiliaryLossData:
    """Auxiliary data returned by total_energy.

    Attributes:
      variance: mean variance over batch, and over all devices if inside a pmap.
      local_energy: local energy for each MCMC configuration.
    """
    variance: jax.Array
    local_energy: jax.Array
    imaginary: Optional[jax.Array] = None
    kinetic: Optional[jax.Array] = None
    potential: Optional[jax.Array] = None
    rashba: Optional[jax.Array] = None
    var_imag: Optional[jax.Array] = None
    var_real: Optional[jax.Array] = None


def make_loss(f, mol, clip_local_energy=5.0, center_at_clip=True, lap_method='scan', max_vmap_batch_size=0, opt_type='kfac'):
    local_energy = hamiltonian.make_local_energy(f, mol, lap_method)
    vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
        folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0, 0))
    batch_network = jax.vmap(f, (None, 0, 0))

    @jax.custom_jvp
    def total_energy(params, batch):
        x, ft = batch
        ke, pe = batch_local_energy(params, x, ft)
        energies = ke + pe
        loss = jnp.mean(energies)
        pmean_loss = constants.pmean_if_pmap(
            loss, axis_name=constants.PMAP_AXIS_NAME)
        variance = jnp.mean((energies)**2) - loss**2
        variance = constants.pmean_if_pmap(variance,
                                           axis_name=constants.PMAP_AXIS_NAME)
        return pmean_loss, AuxiliaryLossData(local_energy=energies,
                                             variance=variance,
                                             )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, (x, ft) = primals
        loss, auxdata = total_energy(params, (x, ft))
        diff = auxdata.local_energy - loss

        primals = (params, x, ft)
        tangents = (tangents[0], tangents[1][0], tangents[1][1])
        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
        primals_out = loss, auxdata
        if opt_type == 'spring':

            tangents_out = jnp.mean(psi_tangent), auxdata
            return primals_out, tangents_out
        else:

            if clip_local_energy > 0.0:
                tv = jnp.mean(jnp.abs(diff))
                tv = constants.pmean_if_pmap(
                    tv, axis_name=constants.PMAP_AXIS_NAME)
                clip_diff = jnp.clip(diff,
                                     -clip_local_energy * tv,
                                     clip_local_energy * tv)
                if center_at_clip:
                    diff_center = jnp.mean(clip_diff)
                    clip_diff = clip_diff - diff_center

            else:
                clip_diff = diff

            kfac_alpha.register_normal_predictive_distribution(
                psi_primal[:, None])

            tangents_out = jnp.mean(clip_diff * psi_tangent), auxdata

            return primals_out, tangents_out

    return total_energy


def make_loss_complex_mol(pf, f, mol, clip_local_energy=5.0, center_at_clip=True, lap_method='scan', max_vmap_batch_size=0, opt_type='kfac'):
    local_energy = hamiltonian.make_energy_complex_mol(pf, mol, lap_method)
    vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
        folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0, 0))
    batch_network = jax.vmap(f, (None, 0, 0))

    @jax.custom_jvp
    def total_energy(params, batch):
        x, ft = batch
        ke, pe = batch_local_energy(params, x, ft)

        energies = ke + pe

        mean_energy = jnp.mean(energies)

        pmean_loss = constants.pmean_if_pmap(
            mean_energy, axis_name=constants.PMAP_AXIS_NAME)

        loss = pmean_loss.real
        imaginary = pmean_loss.imag
        variance = jnp.mean(jnp.abs(energies)**2) - jnp.abs(mean_energy)**2
        var_imag = jnp.mean(jnp.abs(energies.imag)**2) - \
            jnp.abs(mean_energy.imag)**2
        var_real = jnp.mean(jnp.abs(energies.real)**2) - \
            jnp.abs(mean_energy.real)**2
        variance = constants.pmean_if_pmap(
            variance, axis_name=constants.PMAP_AXIS_NAME)
        var_imag = constants.pmean_if_pmap(
            var_imag, axis_name=constants.PMAP_AXIS_NAME)
        var_real = constants.pmean_if_pmap(
            var_real, axis_name=constants.PMAP_AXIS_NAME)
        return loss, AuxiliaryLossData(local_energy=energies,
                                       variance=variance,
                                       imaginary=imaginary,
                                       kinetic=ke,
                                       potential=pe,
                                       var_imag=var_imag,
                                       var_real=var_real,
                                       )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, (x, ft) = primals

        loss, auxdata = total_energy(params, (x, ft))
        diff = auxdata.local_energy - loss

        primals = (params, x, ft)
        tangents = (tangents[0], tangents[1][0], tangents[1][1])
        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)

        conj_psi_tangent = jnp.conjugate(psi_tangent)
        conj_psi_primal = jnp.conjugate(psi_primal)

        primals_out = loss, auxdata
        if opt_type == 'spring':

            term1 = jnp.mean(conj_psi_tangent)
            tangents_out = term1.real, auxdata
            return primals_out, tangents_out
        else:

            if clip_local_energy > 0.0:
                tv_re = jnp.mean(jnp.abs(diff.real))
                tv_im = jnp.mean(jnp.abs(diff.imag))
                tv_re = constants.pmean_if_pmap(
                    tv_re, axis_name=constants.PMAP_AXIS_NAME)
                tv_im = constants.pmean_if_pmap(
                    tv_im, axis_name=constants.PMAP_AXIS_NAME)
                clip_diff_re = jnp.clip(diff.real,
                                        -clip_local_energy * tv_re,
                                        clip_local_energy * tv_re)
                clip_diff_im = jnp.clip(diff.imag,
                                        -clip_local_energy * tv_im,
                                        clip_local_energy * tv_im)
                if center_at_clip:
                    diff_center_re = jnp.mean(clip_diff_re)
                    clip_diff_re = clip_diff_re - diff_center_re
                    diff_center_im = jnp.mean(clip_diff_im)
                    clip_diff_im = clip_diff_im - diff_center_im
                clip_diff = clip_diff_re + clip_diff_im * 1.j
            else:
                clip_diff = diff

            kfac_alpha.register_normal_predictive_distribution(
                (conj_psi_primal)[:, None])

            term1 = jnp.mean(clip_diff * conj_psi_tangent)

            tangents_out = term1.real, auxdata
            return primals_out, tangents_out

    return total_energy


def make_loss_complex(f, mol, clip_local_energy=5.0, center_at_clip=True, lap_method='scan', heg=False, max_vmap_batch_size=0, opt_type='kfac'):
    local_energy = hamiltonian.make_ewald_energy(
        f, mol, lap_method=lap_method, heg=heg)
    vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
        folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0, 0))
    batch_network = jax.vmap(f, (None, 0, 0))

    @jax.custom_jvp
    def total_energy(params, batch):
        x, ft = batch
        ke, ew = batch_local_energy(params, x, ft)
        energies = ke + ew

        mean_energy = jnp.mean(energies)
        pmean_loss = constants.pmean_if_pmap(
            mean_energy, axis_name=constants.PMAP_AXIS_NAME)
        loss = pmean_loss.real
        imaginary = pmean_loss.imag
        variance = jnp.mean(jnp.abs(energies)**2) - jnp.abs(mean_energy)**2
        var_imag = jnp.mean(jnp.abs(energies.imag)**2) - \
            jnp.abs(mean_energy.imag)**2
        var_real = jnp.mean(jnp.abs(energies.real)**2) - \
            jnp.abs(mean_energy.real)**2

        variance = constants.pmean_if_pmap(
            variance, axis_name=constants.PMAP_AXIS_NAME)
        var_imag = constants.pmean_if_pmap(
            var_imag, axis_name=constants.PMAP_AXIS_NAME)
        var_real = constants.pmean_if_pmap(
            var_real, axis_name=constants.PMAP_AXIS_NAME)
        return loss, AuxiliaryLossData(local_energy=energies,
                                       variance=variance,
                                       imaginary=imaginary,
                                       kinetic=ke,
                                       potential=ew,
                                       var_imag=var_imag,
                                       var_real=var_real,
                                       )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, (x, ft) = primals

        loss, auxdata = total_energy(params, (x, ft))
        diff = auxdata.local_energy - loss

        primals = (params, x, ft)
        tangents = (tangents[0], tangents[1][0], tangents[1][1])
        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)

        conj_psi_tangent = jnp.conjugate(psi_tangent)
        conj_psi_primal = jnp.conjugate(psi_primal)

        primals_out = loss, auxdata
        if opt_type == 'spring':
            tangents_out = jnp.mean((conj_psi_tangent).real), auxdata
            return primals_out, tangents_out
        else:

            if clip_local_energy > 0.0:
                tv_re = jnp.mean(jnp.abs(diff.real))
                tv_im = jnp.mean(jnp.abs(diff.imag))
                tv_re = constants.pmean_if_pmap(
                    tv_re, axis_name=constants.PMAP_AXIS_NAME)
                tv_im = constants.pmean_if_pmap(
                    tv_im, axis_name=constants.PMAP_AXIS_NAME)
                clip_diff_re = jnp.clip(diff.real,
                                        -clip_local_energy * tv_re,
                                        clip_local_energy * tv_re)
                clip_diff_im = jnp.clip(diff.imag,
                                        -clip_local_energy * tv_im,
                                        clip_local_energy * tv_im)
                if center_at_clip:
                    diff_center_re = jnp.mean(clip_diff_re)
                    clip_diff_re = clip_diff_re - diff_center_re
                    diff_center_im = jnp.mean(clip_diff_im)
                    clip_diff_im = clip_diff_im - diff_center_im
                clip_diff = clip_diff_re + clip_diff_im * 1.j
            else:
                clip_diff = diff

            kfac_alpha.register_normal_predictive_distribution(
                (conj_psi_primal)[:, None])

            tangents_out = jnp.mean(
                (clip_diff * conj_psi_tangent).real), auxdata
            return primals_out, tangents_out

    return total_energy


def make_loss_complex2d(f, mol, clip_local_energy=5.0, center_at_clip=True, lap_method='scan', use_vmap=False, alpha=1, heg=False, max_vmap_batch_size=0, opt_type='kfac'):
    local_energy = hamiltonian.make_ewald2d_energy(
        f, mol, alpha, lap_method=lap_method, use_vmap=use_vmap, heg=heg)
    vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
        folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0, 0))
    batch_network = jax.vmap(f, (None, 0, 0))

    @jax.custom_jvp
    def total_energy(params, batch):
        x, ft = batch
        ke, ew, rashba = batch_local_energy(params, x, ft)
        energies = ke + ew + rashba

        mean_energy = jnp.mean(energies)
        pmean_loss = constants.pmean_if_pmap(
            mean_energy, axis_name=constants.PMAP_AXIS_NAME)
        loss = pmean_loss.real
        imaginary = pmean_loss.imag
        variance = jnp.mean(jnp.abs(energies)**2) - jnp.abs(mean_energy)**2
        var_imag = jnp.mean(jnp.abs(energies.imag)**2) - \
            jnp.abs(mean_energy.imag)**2
        var_real = jnp.mean(jnp.abs(energies.real)**2) - \
            jnp.abs(mean_energy.real)**2

        variance = constants.pmean_if_pmap(
            variance, axis_name=constants.PMAP_AXIS_NAME)
        var_imag = constants.pmean_if_pmap(
            var_imag, axis_name=constants.PMAP_AXIS_NAME)
        var_real = constants.pmean_if_pmap(
            var_real, axis_name=constants.PMAP_AXIS_NAME)
        return loss, AuxiliaryLossData(local_energy=energies,
                                       variance=variance,
                                       imaginary=imaginary,
                                       rashba=rashba,
                                       kinetic=ke,
                                       potential=ew,
                                       var_imag=var_imag,
                                       var_real=var_real,
                                       )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, (x, ft) = primals

        loss, auxdata = total_energy(params, (x, ft))
        diff = auxdata.local_energy - loss

        primals = (params, x, ft)
        tangents = (tangents[0], tangents[1][0], tangents[1][1])
        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)

        conj_psi_tangent = jnp.conjugate(psi_tangent)
        conj_psi_primal = jnp.conjugate(psi_primal)

        primals_out = loss, auxdata
        if opt_type == 'spring':

            tangents_out = jnp.mean((conj_psi_tangent).real), auxdata
            return primals_out, tangents_out
        else:

            if clip_local_energy > 0.0:
                tv_re = jnp.mean(jnp.abs(diff.real))
                tv_im = jnp.mean(jnp.abs(diff.imag))

                tv_re = constants.pmean_if_pmap(
                    tv_re, axis_name=constants.PMAP_AXIS_NAME)
                tv_im = constants.pmean_if_pmap(
                    tv_im, axis_name=constants.PMAP_AXIS_NAME)

                clip_diff_re = jnp.clip(diff.real,
                                        -clip_local_energy * tv_re,
                                        clip_local_energy * tv_re)
                clip_diff_im = jnp.clip(diff.imag,
                                        -clip_local_energy * tv_im,
                                        clip_local_energy * tv_im)
                if center_at_clip:
                    diff_center_re = jnp.mean(clip_diff_re)
                    clip_diff_re = clip_diff_re - diff_center_re
                    diff_center_im = jnp.mean(clip_diff_im)
                    clip_diff_im = clip_diff_im - diff_center_im
                clip_diff = clip_diff_re + clip_diff_im * 1.j
            else:
                clip_diff = diff

            kfac_alpha.register_normal_predictive_distribution(
                (conj_psi_primal)[:, None])

            tangents_out = jnp.mean(
                (clip_diff * conj_psi_tangent).real), auxdata
            return primals_out, tangents_out
    return total_energy
