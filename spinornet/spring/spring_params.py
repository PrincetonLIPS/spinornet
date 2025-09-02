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

import jax
import chex
import jax.numpy as jnp
from spinornet import constants


def _update_metrics_with_noclip(
    energy_noclip: float, variance_noclip: float, metrics
):
    if energy_noclip is not None:
        metrics.update({"energy_noclip": energy_noclip})
    if variance_noclip is not None:
        metrics.update({"variance_noclip": variance_noclip})
    return metrics


def _make_traced_fn_with_single_metrics(
    update_param_fn,
    apply_pmap: bool,
    metrics_to_get_first=None,
):
    if not apply_pmap:
        return jax.jit(update_param_fn)

    pmapped_update_param_fn = constants.pmap(update_param_fn)

    def pmapped_update_param_fn_with_single_metrics(params, optimizer_state, key, pos, feat):
        params, optimizer_state, metrics = pmapped_update_param_fn(
            params, optimizer_state, key, pos, feat
        )
        if metrics_to_get_first is None:
            metrics = constants.get_first(metrics)
        else:
            for metric in metrics_to_get_first:
                distributed_metric = metrics.get(metric)
                if distributed_metric is not None:
                    metrics[metric] = constants.get_first(distributed_metric)

        return params, optimizer_state, metrics

    return pmapped_update_param_fn_with_single_metrics


def tree_reduce_l1(xs) -> chex.Numeric:
    """L1 norm of a pytree as a flattened vector."""
    concat_xs, _ = jax.flatten_util.ravel_pytree(xs)
    return jnp.sum(jnp.abs(concat_xs))


def create_grad_energy_update_param_fn(
    energy_data_val_and_grad,
    optimizer_apply,
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
):
    """Create the `update_param_fn` based on the gradient of the total energy.

    See :func:`~vmcnet.train.vmc.vmc_loop` for its usage.

    Args:
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. 
        optimizer_apply (Callable): applies an update to the parameters. 
        apply_pmap (bool, optional): whether to apply jax.pmap to the walker function.
            If False, applies jax.jit. Defaults to True.

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. 
        The function is pmapped if apply_pmap is True, and jitted if apply_pmap is
        False.
    """

    def update_param_fn(params, optimizer_state, key, pos, feat, agespos, agesspin):
        if isinstance(optimizer_state, dict) and len(optimizer_state) == 1 and "single_element" in optimizer_state:
            optimizer_state = optimizer_state["single_element"]
        key, subkey = jax.random.split(key)

        energy_data, grad_energy = energy_data_val_and_grad(
            params, (pos, feat))
        energy, aux_energy_data = energy_data

        params, optimizer_state = optimizer_apply(
            params,
            optimizer_state,
            (pos, feat),
            dict(
                centered_local_energies=aux_energy_data["local_energy"] - energy),
        )

        metrics = {"loss": energy, "aux": aux_energy_data, "agespos": agespos,
                   "agesspin": agesspin}

        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        optimizer_state = {"single_element": optimizer_state}
        return params, optimizer_state, metrics

    if apply_pmap:
        update_param_fn = constants.pmap(update_param_fn)
    else:
        update_param_fn = jax.jit(update_param_fn)

    return update_param_fn
