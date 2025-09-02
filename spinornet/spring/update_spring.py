# This code was modified from vmcnet (https://github.com/jeffminlin/vmcnet)

# MIT License

# Copyright (c) 2021

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file may have been modified by Spinornet authors.
# All Spinornet modifications are Copyright 2025 Spinornet authors.

import chex
import jax.numpy as jnp
import jax
from spinornet import constants


def tree_inner_product(tree1, tree2):
    """Inner product of two pytrees with the same structure."""
    leaf_inner_prods = jax.tree_map(lambda a, b: jnp.sum(a * b), tree1, tree2)
    return jnp.sum(jax.flatten_util.ravel_pytree(leaf_inner_prods)[0])


def multiply_tree_by_scalar(tree, scalar: chex.Numeric):
    """Multiply all leaves of a pytree by a scalar."""
    return jax.tree_map(lambda x: scalar * x, tree)


def get_spring_update_fn(
    log_psi_apply,
    energy_data_val_and_grad,
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
    momentum: chex.Scalar = 0.0,
    clip: chex.Scalar = 5.0,
    center_at_clip: bool = True,
):
    """
    Get the SPRING update function.

    Args:
        log_psi_apply (Callable): computes log(psi(x)), where the signature of this
            function is (params, x, ft) -> log(psi(x,ft))
        damping (float): damping parameter
        mu (float): SPRING-specific regularization

    Returns:
        Callable: SPRING update function. 
    """

    venergy_data_val_and_grad = jax.vmap(
        energy_data_val_and_grad, (None, (0, 0)))

    def raveled_log_psi_grad(log_grads):
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(0))

    def spring_update_fn(

        energies,
        params,
        prev_grad,
        data,
    ):
        positions, feats = data
        nchains = positions.shape[0]

        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad_decayed = mu * prev_grad

        _, log_psi_grads = venergy_data_val_and_grad(
            params, (positions[:, None, ...], feats[:, None, ...]))

        log_psi_grads = batch_raveled_log_psi_grad(
            log_psi_grads) / jnp.sqrt(nchains)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)

        T = Ohat @ Ohat.T
        ones = jnp.ones((nchains, 1))

        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)

        diff = energies

        if clip > 0.0:
            tv_re = jnp.mean(jnp.abs(diff.real))
            tv_im = jnp.mean(jnp.abs(diff.imag))

            tv_re = constants.pmean_if_pmap(
                tv_re, axis_name=constants.PMAP_AXIS_NAME)
            tv_im = constants.pmean_if_pmap(
                tv_im, axis_name=constants.PMAP_AXIS_NAME)

            clip_diff_re = jnp.clip(diff.real,
                                    -clip * tv_re,
                                    clip * tv_re)
            clip_diff_im = jnp.clip(diff.imag,
                                    -clip * tv_im,
                                    clip * tv_im)
            if center_at_clip:
                diff_center_re = jnp.mean(clip_diff_re)
                clip_diff_re = clip_diff_re - diff_center_re
                diff_center_im = jnp.mean(clip_diff_im)
                clip_diff_im = clip_diff_im - diff_center_im
            clip_diff = clip_diff_re + clip_diff_im * 1.j
        else:
            clip_diff = diff

        epsilon_bar = clip_diff / jnp.sqrt(nchains)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed
        # epsion_tilde.real to avoid complex warning.
        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(
            T_reg, epsion_tilde.real, assume_a="pos"
        )

        SR_G = dtheta_residual + prev_grad_decayed
        SR_G = (1 - momentum) * SR_G + momentum * prev_grad

        return unravel_fn(SR_G)

    return spring_update_fn


def constrain_norm(
    grad,
    norm_constraint: chex.Numeric = 0.001,
):
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = constants.pmean_if_pmap_spring(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
