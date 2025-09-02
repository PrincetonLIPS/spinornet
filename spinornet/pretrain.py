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
import optax
from absl import logging
import jax
from jax import grad
import numpy as np
import spinornet.kfac_alpha as kfac_alpha
from spinornet import curvature_tags_and_blocks
from spinornet import constants
from spinornet import init_guess
from spinornet.kfac_alpha._src import kfac_utils
from spinornet.network import logdet_matmul


def pretrain_using_net(sharded_key, nnwf, hartree_fock, pos, feats, agespos, agesspin, mol, params, hfparams, mcmc_step_both, cfg, learning_rate_schedule, shared_mom, shared_damping, mcmc_width, mcmc_spin_width):
    init_spin_width = cfg.mcmc.init_spin_width
    local_batch_size = cfg.batch_size
    learning_rate = cfg.pretrain.lr
    numiter = cfg.pretrain.iterations
    optimizer = cfg.pretrain.optimizer

    def loss_fn(params, batch) -> jnp.ndarray:
        """MSE loss."""
        pos, feat, hforb = batch
        pred = nnwf(params, pos, feat)
        kfac_alpha.register_squared_error_loss(
            jnp.concatenate((pred.real, pred.imag)))
        result = jnp.mean(
            (pred.real - hforb[:, :, :, None].real)**2 + (pred.imag - hforb[:, :, :, None].imag)**2)
        return constants.pmean_if_pmap(result, axis_name=constants.PMAP_AXIS_NAME)

    if optimizer == 'adam':
        optimizer = optax.adam(learning_rate)
        opt_state = constants.pmap(optimizer.init)(params)

        @jax.jit
        def update_step(params, opt_state, key, pos, feat, hforb):
            del key
            loss, gradients = jax.value_and_grad(
                loss_fn, argnums=0, has_aux=False)(params, (pos, feat, hforb))
            gradients = constants.pmean_if_pmap(
                gradients, axis_name=constants.PMAP_AXIS_NAME)
            updates, new_opt_state = optimizer.update(gradients, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        update_step = constants.pmap(update_step)

    elif optimizer == 'kfac':
        val_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=False)
        optimizer = kfac_alpha.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=False,
            value_func_has_rng=False,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode='fisher_gradients',
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME,
            auto_register_kwargs=dict(
                graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
            ),
            iscomplex=0
        )
        sharded_key, subkeys = kfac_utils.p_split(sharded_key)
        hartree_fock = constants.pmap(hartree_fock)
        hforb = hartree_fock(params, pos, feats)

        opt_state = optimizer.init(params, subkeys, (pos, feats, hforb))

        def update_step(params, opt_state, key, pos, feat, hforb):
            params, opt_state, stats = optimizer.step(
                params, opt_state, key, batch=(pos, feat, hforb),  global_step_int=i, damping=shared_damping, momentum=shared_mom)
            return params, opt_state, stats['loss']

    for i in range(numiter):
        sharded_key, subkeys = constants.p_split(sharded_key)

        pos, feats, pmovepos, pmovespin, agespos, agesspin = mcmc_step_both(
            params, pos, feats, subkeys, mcmc_width, mcmc_spin_width, agespos, agesspin)

        hforb = hartree_fock(hfparams, pos, feats)

        sharded_key, subkeys = constants.p_split(sharded_key)
        params, opt_state, loss = update_step(
            params, opt_state, subkeys, pos, feats, hforb)

        if i % 1 == 0:

            logging.info('Pretrain iter %05d: Loss=%03.6f',
                         i, loss[0]
                         )

    return params, opt_state, pos, feats, agespos, agesspin
