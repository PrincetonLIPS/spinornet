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

import optax
from spinornet.spring.update_spring import get_spring_update_fn, constrain_norm as constrain_norm_spring
from spinornet.spring.spring_params import create_grad_energy_update_param_fn
from spinornet import constants


def _init_optax_optimizer(
    optimizer: optax.GradientTransformation, params, apply_pmap: bool = True
) -> optax.OptState:
    optimizer_init = optimizer.init
    if apply_pmap:
        optimizer_init = constants.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def get_spring_update_fn_and_state(
    log_psi_apply,
    params,
    energy_data_val_and_grad,
    learning_rate_schedule,
    optimizer_config,
    clip,
    center_at_clip,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
):
    """Get an update param function and initial state for SPRING.

    Args:
        log_psi_apply (Callable): computes log(psi(x)), where the signature of this
            function is (params, x, ft) -> log(psi(x,ft))
        params (pytree): params with which to initialize optimizer state
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. 
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    spring_update_fn = get_spring_update_fn(
        log_psi_apply,
        energy_data_val_and_grad,
        optimizer_config.damping,
        optimizer_config.mu,
        optimizer_config.momentum,
        clip,
        center_at_clip
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(params, optimizer_state, data, aux):
        grad = spring_update_fn(
            aux["centered_local_energies"],
            params,
            prev_update(optimizer_state),
            data,
        )

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )

        if optimizer_config.constrain_norm:
            updates = constrain_norm_spring(
                updates,
                optimizer_config.norm_constraint,
            )

        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state
