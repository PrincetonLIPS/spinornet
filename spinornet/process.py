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

import ml_collections
import time
import pyqmc.api as pyq
from spinornet import hf
from spinornet import init_guess
import jax
import optax
from spinornet import network
import jax.numpy as jnp
import numpy as np
import haiku as hk
from jax import vmap, grad
from spinornet import pretrain
from spinornet import checkpoint
from spinornet import hamiltonian
from spinornet import train
from spinornet import constants
from absl import logging
from spinornet import mcmc
from spinornet import mcmcl
from spinornet.utils import writers
import spinornet.kfac_alpha as kfac_alpha
from spinornet import curvature_tags_and_blocks
import os
from functools import partial
from spinornet import distance
from spinornet.kfac_alpha._src import kfac_utils
from spinornet import spins_jax
from spinornet.spring import spring_opt
from jax import lax


def unpmap(wpmap):
    wu = wpmap.copy()
    for key in list(wpmap.keys()):
        for i, key2 in enumerate(wpmap[key]):
            try:
                wu[key][key2] = jax.tree_map(lambda x: x[0], wpmap[key][key2])
            except:
                wu[key] = jax.tree_map(lambda x: x[0], wpmap[key])
    return wu


def map_nested_fn(fn):
    '''Recursively apply `fn` to the key-value pairs of a nested dict'''
    def map_fn(nested_dict):
        return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()}
    return map_fn


def get_params_initialization_key(deterministic):
    '''
    The key point here is to make sure different hosts uses the same RNG key
    to initialize network parameters.
    '''
    if deterministic:
        seed = 888
    else:

        # The overly complicated action here is to make sure different hosts get
        # the same seed.
        @constants.pmap
        def average_seed(seed_array):
            return jax.lax.pmean(jnp.mean(seed_array), axis_name=constants.PMAP_AXIS_NAME)

        local_seed = time.time()
        float_seed = average_seed(
            jnp.ones(jax.local_device_count()) * local_seed)[0]
        seed = int(1e6 * float_seed)
    print(f'params initialization seed: {seed}')
    return jax.random.PRNGKey(seed)


def process(cfg: ml_collections.ConfigDict):

    mol = cfg.system.pyscf_cell

    if cfg.system.type == 'mol':
        hartree_fock = hf.SCFMol(mol)
        hartree_fock.run()

    if cfg.system.type == 'solid':
        hartree_fock = hf.SCF(cell=mol, twist=cfg.network.twist)
        hartree_fock.init_scf()

    if cfg.system.type == 'heg2d':
        hartree_fock = hf.HEGscf2d(cell=mol, twist=cfg.network.twist)

    num_hosts, host_idx = 1, 0

    # Device logging
    print('total devices: '+f'{jax.device_count()}')
    num_devices = jax.device_count()
    local_batch_size = cfg.batch_size // num_hosts
    logging.info('Starting QMC with %i XLA devices', num_devices)
    if local_batch_size % num_devices != 0:
        raise ValueError('Batch size must be divisible by number of devices, '
                         'got batch size {} for {} devices.'.format(
                             local_batch_size, num_devices))

    seed = int(1e6 * time.time())
    logging.info(f'seed: {seed}')
    # change it to get different initializations
    key = jax.random.PRNGKey(seed)

    data_shape = (num_devices, local_batch_size // num_devices)
    key, splitkey = jax.random.split(key)
    datainit = hartree_fock.generate_wf_samples_jax(
        splitkey, local_batch_size, cfg.mcmc.init_config_width)
    feats_inits_style = {'random': init_guess.init_spins_random,
                         'half': init_guess.init_spins_half,
                         'nelec': init_guess.init_spins_nelec,
                         }

    key, splitkey = jax.random.split(key)
    if cfg.init_spin_type == 'half':
        featsinits = feats_inits_style[cfg.init_spin_type](
            splitkey, datainit, mol, sigma=1e-10)

    elif cfg.init_spin_type == 'nelec':
        featsinits = feats_inits_style[cfg.init_spin_type](
            splitkey, datainit, mol, mol.nelec)
    else:
        featsinits = feats_inits_style[cfg.init_spin_type](
            splitkey, datainit, mol, sigma=cfg.mcmc.init_spin_width)

    datainit = jnp.reshape(datainit, data_shape + datainit.shape[1:])
    datainit = constants.broadcast_all_local_devices(datainit)

    featsinits = jnp.reshape(featsinits, data_shape + featsinits.shape[1:])
    featsinits = constants.broadcast_all_local_devices(featsinits)

    lattice = np.array([[100., 0, 0,], [0, 100., 0], [0, 0, 120.]])
    klist = None
    if cfg.pbc and cfg.system.type == 'solid':
        lattice = mol.a
        klist = hartree_fock.klist

    if cfg.pbc and cfg.system.type == 'heg2d':
        lattice = mol.a
        klist = mol.klist

    network_kwargs = {
        'ferminet': cfg.network.ferminet,
        'spinornet': cfg.network.spinornet,
        'spinornet_single': cfg.network.spinornet_single,
        'spinornet_pbc': cfg.network.spinornet_pbc,
        'spinornet_pbc2dHEG': cfg.network.spinornet_pbc2dHEG,

    }

    system_dict = {
        'simulation_cell': mol,
        'klist': klist,
        'lattice': lattice,
        'hartree_fock': hartree_fock,
        'ndets': cfg.network.ndets,
        'network_method': cfg.network.network_type,
        'kwargs': network_kwargs[cfg.network.network_type]
    }
    slater_mat = network.make_nn_psi(
        **system_dict, hf_method=cfg.network.hf_method, method_name='eval_mats', alpha=cfg.soc_alpha)
    slater_slogdet = network.make_nn_psi(
        **system_dict, hf_method=cfg.network.hf_method, method_name='eval_slogdet', alpha=cfg.soc_alpha)
    slater_logdet = network.make_nn_psi(
        **system_dict, hf_method=cfg.network.hf_method, method_name='eval_logdet', alpha=cfg.soc_alpha)

    batch_network = vmap(slater_slogdet.apply, (None, 0, 0))
    batch_val_grad_slogdet = vmap(jax.value_and_grad(
        slater_slogdet.apply, argnums=1), (None, 0, 0))
    batch_network_logdet = vmap(slater_logdet.apply, (None, 0, 0))

    params_initialization_key = get_params_initialization_key(
        cfg.debug.deterministic)
    params = jax.jit(slater_mat.init)(
        params_initialization_key, datainit[0][0], featsinits[0][0])

    slater_mat = vmap(slater_mat.apply, (None, 0, 0))

    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)

    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

    shared_t = constants.replicate_all_local_devices(jnp.zeros([]))
    shared_mom = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_utils.replicate_all_local_devices(
        jnp.asarray(cfg.optim.kfac.damping))
    sharded_key = constants.make_different_rng_key_on_all_devices(key)

    if not cfg.complex:
        eval_loss = train.make_loss(slater_slogdet.apply, mol, cfg.optim.clip_local_energy, cfg.optim.center_at_clip,
                                    cfg.optim.lap_method, max_vmap_batch_size=cfg.optim.max_vmap_batch_size, opt_type=cfg.optim.optimizer)
    if cfg.complex and cfg.system.type == 'mol':
        eval_loss = train.make_loss_complex_mol(slater_logdet.apply, slater_logdet.apply, mol, cfg.optim.clip_local_energy,
                                                cfg.optim.center_at_clip, cfg.optim.lap_method, max_vmap_batch_size=cfg.optim.max_vmap_batch_size, opt_type=cfg.optim.optimizer)
    if cfg.complex and cfg.system.type == 'solid':
        eval_loss = train.make_loss_complex(slater_logdet.apply, mol, cfg.optim.clip_local_energy, cfg.optim.center_at_clip,
                                            cfg.optim.lap_method, max_vmap_batch_size=cfg.optim.max_vmap_batch_size, opt_type=cfg.optim.optimizer)
    if cfg.complex and cfg.system.type == 'heg2d':
        eval_loss = train.make_loss_complex2d(slater_logdet.apply, mol, cfg.optim.clip_local_energy, cfg.optim.center_at_clip, cfg.optim.lap_method,
                                              cfg.optim.lap_use_vmap, alpha=cfg.soc_alpha, heg=True, max_vmap_batch_size=cfg.optim.max_vmap_batch_size, opt_type=cfg.optim.optimizer)

    if cfg.mcmc.pos_type == 'gaussian':
        mcmc_step_pos = mcmc.make_mcmc_step(batch_network,
                                            local_batch_size//jax.local_device_count(), steps=1, pbc=cfg.pbc, lattice=lattice, reciplattice=jnp.linalg.inv(lattice), is2d=cfg.is2d)

    elif cfg.mcmc.pos_type == 'langevin':
        mcmc_step_pos = mcmcl.make_mcmc_step(batch_val_grad_slogdet,
                                             local_batch_size//jax.local_device_count(), steps=1, pbc=cfg.pbc, lattice=lattice, reciplattice=jnp.linalg.inv(lattice), is2d=cfg.is2d)

    if cfg.mcmc.spin_type == 'cont':
        mcmc_step_spin = mcmc.make_mcmc_step_spin(slater_mat,
                                                  local_batch_size//jax.local_device_count(),  steps=1)

    elif cfg.mcmc.spin_type == 'discrete':
        mcmc_step_spin = mcmc.make_mcmc_step_spin_discrete(slater_mat,
                                                           local_batch_size//jax.local_device_count(), steps=1)

    train_schema = ['step', 'loss', 'error', 'sigma', 'variance', 'imaginary', 'kinetic_re', 'kinetic_im', 'potential', 'rashba_re',
                    'rashba_im', 'var_imag', 'var_real', 'pmove', 'pmovepos', 'pmovespin', 'mcmc_width', 'mcmc_spin_width', 'max_age_pos', 'max_age_spin']
    writer_manager = writers.Writer(
        name='train_stats',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False)

    pos = datainit

    t_init = 0
    feats = featsinits
    agespos = jnp.zeros(shape=datainit.shape[0:2])
    agesspin = jnp.zeros(shape=datainit.shape[0:2])
    mcmc_width = constants.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.init_width))
    mcmc_spin_width = constants.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.init_spin_width))

    # if loading from checkpoint:
    if cfg.log.restore_path != '':
        paramsload = checkpoint.restore_params(
            cfg.log.restore_path, cfg.log.restore_epoch)
        t_init, pos, feats, opt_state, mcmc_width, mcmc_spin_width, _, _ = checkpoint.restore(os.path.join(
            cfg.log.restore_path, f'qmcjax_ckpt_{int(cfg.log.restore_epoch):06d}.npz'), shape_check=False)

        upmapp = unpmap(paramsload)

        params = constants.replicate_all_local_devices(upmapp)
        mcmc_width = constants.replicate_all_local_devices(
            jnp.asarray(mcmc_width[0]))
        mcmc_spin_width = constants.replicate_all_local_devices(
            jnp.asarray(mcmc_spin_width[0]))
        pos = pos.reshape((-1,) + datainit.shape[2:])

        posrs, _ = jax.vmap(distance.enforce_pbc, (None, None, 0))(
            lattice, jnp.linalg.inv(lattice), pos)
        posrs = posrs.reshape(data_shape + datainit.shape[2:])
        if cfg.mcmc.spin_sample:
            featsrs = feats.reshape(
                (-1,) + featsinits.shape[2:])[:cfg.batch_size].reshape(data_shape + featsinits.shape[2:])
            feats = constants.broadcast_all_local_devices(featsrs)
        else:
            feats = featsinits
        pos = constants.broadcast_all_local_devices(posrs)
    else:
        params = constants.replicate_all_local_devices(params)

    if cfg.mcmc.spin_sample:

        def mcmc_step_both(params, pos, feats, key, mcmc_width, mcmc_spin_width, agespos, agesspin):

            def step_fn(i, x):
                pos, feats, key, _, _, agespos, agesspin = x

                key, subkey = jax.random.split(key)
                pos, feats, pmovepos, agespos = mcmc_step_pos(
                    params, pos, feats, subkey, mcmc_width, mcmc_spin_width, agespos)

                key, subkey = jax.random.split(key)
                pos, feats, pmovespin, agesspin = mcmc_step_spin(
                    params, pos, feats, subkey, mcmc_width, mcmc_spin_width, agesspin)

                return pos, feats, key, pmovepos, pmovespin, agespos, agesspin

            pos, feats, key, pmovepos, pmovespin, agespos, agesspin = lax.fori_loop(
                0, cfg.mcmc.steps, step_fn, (pos, feats,
                                             key, 0.0, 0.0, agespos, agesspin)
            )
            return pos, feats, pmovepos, pmovespin, agespos, agesspin
        mcmc_step_both = constants.pmap(mcmc_step_both)

    else:
        mcmc_spin_width = constants.replicate_all_local_devices(
            jnp.asarray(0.))
        pmovespin = 1

        def mcmc_step_both(params, pos, feats, key, mcmc_width, mcmc_spin_width, agespos, agesspin):
            def step_fn(i, x):
                pos, feats, key, _, agespos = x
                nkey, subkey = jax.random.split(key)
                pos, feats, pmovepos, agespos = mcmc_step_pos(
                    params, pos, feats, subkey, mcmc_width, mcmc_spin_width, agespos)
                return pos, feats, nkey, pmovepos, agespos

            pos, feats, key, pmovepos, agespos = lax.fori_loop(
                0, cfg.mcmc.steps, step_fn, (pos, feats, key, 0.0, agespos)
            )
            return pos, feats, pmovepos, pmovespin, agespos, agesspin
        mcmc_step_both = constants.pmap(mcmc_step_both)

    sharded_key, subkeys = constants.p_split(sharded_key)
    if cfg.pretrain.iterations > 0 and cfg.optim.optimizer != 'none' and cfg.log.restore_path == '':
        slater_hf_orb = network.make_nn_psi(
            **system_dict, hf_method=cfg.pretrain.hf_method, method_name='eval_hf_orb', alpha=cfg.soc_alpha)
        hfparams = jax.jit(slater_hf_orb.init)(
            params_initialization_key, datainit[0][0], featsinits[0][0])
        hfparams = constants.replicate_all_local_devices(hfparams)
        slater_hf_orb = vmap(slater_hf_orb.apply, (None, 0, 0))
        params, opt_state, pos, feats, agespos, agesspin = pretrain.pretrain_using_net(
            subkeys, slater_mat, slater_hf_orb, pos, feats, agespos, agesspin, mol, params, hfparams, mcmc_step_both, cfg, learning_rate_schedule,  shared_mom, shared_damping, mcmc_width, mcmc_spin_width)

        checkpoint.save_params(ckpt_save_path, params, -1)

    if cfg.optim.optimizer == 'adam':

        optimizer = optax.chain(optax.scale_by_adam(**cfg.optim.adam),
                                optax.scale_by_schedule(
                                    learning_rate_schedule),
                                optax.scale(-1.))

        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=cfg.optim.ministeps)
        opt_state = jax.pmap(optimizer.init)(params)

        def update_step(params, opt_state, key, pos, feat, agespos, agesspin):
            del key
            (loss, auxdata), gradients = jax.value_and_grad(
                eval_loss, argnums=0, has_aux=True)(params, (pos, feat))
            gradients = constants.pmean_if_pmap(gradients,
                                                axis_name=constants.PMAP_AXIS_NAME)
            updates, new_opt_state = optimizer.update(gradients, opt_state)

            new_params = optax.apply_updates(params, updates)
            stats = {}
            stats['loss'] = loss
            stats['aux'] = auxdata
            stats['agespos'] = agespos
            stats['agesspin'] = agesspin
            return new_params, new_opt_state, stats
        update_step = constants.pmap(update_step)

    elif cfg.optim.optimizer == 'kfac':
        val_and_grad = jax.value_and_grad(eval_loss, argnums=0, has_aux=True)
        optimizer = kfac_alpha.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            value_func_has_rng=False,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode='fisher_exact',
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME,
            auto_register_kwargs=dict(
                graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
            ),
            iscomplex=cfg.complex,
            include_norms_in_stats=True,
            include_per_param_norms_in_stats=True,
            # use_adaptive_damping=True,
            # initial_damping=0.001,
        )

        sharded_key, subkeys = kfac_utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, (datainit, featsinits))
        opt_state.step_counter = t_init

        def update_step(params, opt_state, key, pos, feat, agespos, agesspin):
            params, opt_state, stats = optimizer.step(
                params, opt_state, key, batch=(pos, feat),  global_step_int=i, damping=shared_damping, momentum=shared_mom)
            stats['agespos'] = agespos
            stats['agesspin'] = agesspin
            return params, opt_state, stats

    elif cfg.optim.optimizer == 'spring':
        val_and_grad = jax.value_and_grad(eval_loss, argnums=0, has_aux=True)
        update_step, opt_state = spring_opt.get_spring_update_fn_and_state(
            slater_logdet.apply, params, val_and_grad, learning_rate_schedule, cfg.optim.spring, cfg.optim.clip_local_energy, cfg.optim.center_at_clip, apply_pmap=True)

    elif cfg.optim.optimizer == 'none' and not cfg.sampling_only:
        optimizer = None
        opt_state = None

        def update_step(params, opt_state, key, pos, feat, agespos, agesspin):
            """Evaluates just the loss."""
            loss, aux_data = eval_loss(params, (pos, feat))
            stats = {}
            stats['loss'] = loss
            stats['aux'] = aux_data
            stats['agespos'] = agespos
            stats['agesspin'] = agesspin
            return params, opt_state, stats
        update_step = constants.pmap(update_step)
    elif cfg.sampling_only == True:
        opt_state = None

        def update_step(params, opt_state, key, pos, feat, agespos, agesspin):
            """No evaluation."""
            stats = {}
            stats['loss'] = 0
            stats['aux'] = 0
            return params, opt_state, stats
        update_step = constants.pmap(update_step)

    curr_batch_size = cfg.batch_size
    with writer_manager as writer:
        for i in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = constants.p_split(sharded_key)
            pos, feats, pmovepos, pmovespin, agespos, agesspin = mcmc_step_both(
                params, pos, feats, subkeys, mcmc_width, mcmc_spin_width, agespos, agesspin)

        if cfg.optim.optimizer == 'kfac' or cfg.optim.optimizer == 'adam' or cfg.optim.optimizer == 'spring':
            loss, auxdata = constants.pmap(eval_loss)(params, (pos, feats))

        elif cfg.optim.optimizer == 'none':
            loss, auxdata = constants.pmap(eval_loss)(params, (pos, feats))

            checkpoint.save_params(ckpt_save_path, params, -1)
            writer.write(
                -1,
                step=-1,
                loss=loss[0],
                error=np.sqrt(auxdata.variance[0] / (local_batch_size - 1)),
                sigma=np.sqrt(auxdata.variance[0]),
                variance=auxdata.variance[0],
                pmovepos=pmovepos[0],
                pmovespin=pmovespin[0],
                mcmc_width=mcmc_width[0],
                mcmc_spin_width=mcmc_spin_width[0])
        logging.info('MCMC Burnin: Loss=%03.6f, variance=%03.4f,  pmovepos=%0.2f, pmovespin=%0.3f',
                     loss[0], auxdata.variance[0], pmovepos[0], pmovespin[0]
                     )

        curr_batch_size = cfg.batch_size
        for i in range(t_init, t_init + cfg.optim.iterations):
            sharded_key, subkeys = constants.p_split(sharded_key)
            pos, feats, pmovepos, pmovespin, agespos, agesspin = mcmc_step_both(
                params, pos, feats, subkeys, mcmc_width, mcmc_spin_width, agespos, agesspin)
            sharded_key, subkeys = constants.p_split(sharded_key)
            params, opt_state, stats = update_step(
                params, opt_state, subkeys, pos, feats, agespos, agesspin)

            loss = stats['loss']
            auxdata = stats['aux']

            shared_t = shared_t + 1
            if i % 1 == 0:
                if cfg.complex and cfg.soc_alpha != 0 and not cfg.sampling_only:
                    logging.info(f'Energy training iter {i:05d}: Loss={loss[0]:03.6f}, variance={auxdata.variance[0]:03.4f}, imaginary={auxdata.imaginary[0]:03.6f}, kinetic={jnp.mean(auxdata.kinetic):.4f}, potential={jnp.mean(auxdata.potential):03.4f}, rashba={jnp.mean(auxdata.rashba):.4f},pmovepos={pmovepos[0]:0.2f}, pmovespin={jnp.mean(pmovespin):0.3f}, mcmc_width={mcmc_width[0]:0.2f}, max_age_pos={jnp.max(agespos)}, max_age_spin={jnp.max(agesspin)}',
                                 )
                    writer.write(
                        i,
                        step=i,
                        loss=loss[0],
                        error=np.sqrt(
                            auxdata.variance[0] / (local_batch_size - 1)),
                        sigma=np.sqrt(auxdata.variance[0]),
                        variance=auxdata.variance[0],
                        imaginary=auxdata.imaginary[0],
                        kinetic_re=jnp.mean(auxdata.kinetic.real),
                        kinetic_im=jnp.mean(auxdata.kinetic.imag),
                        potential=jnp.mean(auxdata.potential),
                        rashba_re=jnp.mean(auxdata.rashba.real),
                        rashba_im=jnp.mean(auxdata.rashba.imag),
                        var_imag=jnp.mean(auxdata.var_imag),
                        var_real=jnp.mean(auxdata.var_real),
                        pmovepos=pmovepos[0],
                        pmovespin=pmovespin[0],
                        mcmc_width=mcmc_width[0],
                        mcmc_spin_width=mcmc_spin_width[0],
                        max_age_pos=jnp.max(agespos),
                        max_age_spin=jnp.max(agesspin),
                    )
                elif cfg.complex and not cfg.sampling_only:
                    logging.info(f'Energy training iter {i:05d}: Loss={loss[0]:03.6f}, variance={auxdata.variance[0]:03.4f}, imaginary={auxdata.imaginary[0]:03.6f}, kinetic={jnp.mean(auxdata.kinetic):.4f}, potential={jnp.mean(auxdata.potential):03.4f},  pmovepos={pmovepos[0]:0.2f}, pmovespin={jnp.mean(pmovespin):0.3f}, mcmc_width={mcmc_width[0]:.2f}, mcmc_spin_width={mcmc_spin_width[0]:.3f}, max_age_pos={jnp.max(agespos)}, max_age_spin={jnp.max(agesspin)}',
                                 )
                    writer.write(
                        i,
                        step=i,
                        loss=loss[0],
                        error=np.sqrt(
                            auxdata.variance[0] / (local_batch_size - 1)),
                        sigma=np.sqrt(auxdata.variance[0]),
                        variance=auxdata.variance[0],
                        imaginary=auxdata.imaginary[0],
                        kinetic_re=jnp.mean(auxdata.kinetic.real),
                        kinetic_im=jnp.mean(auxdata.kinetic.imag),
                        potential=jnp.mean(auxdata.potential),
                        var_imag=jnp.mean(auxdata.var_imag),
                        var_real=jnp.mean(auxdata.var_real),
                        pmovepos=pmovepos[0],
                        pmovespin=pmovespin[0],
                        mcmc_width=mcmc_width[0],
                        mcmc_spin_width=mcmc_spin_width[0],
                        max_age_pos=jnp.max(agespos),
                        max_age_spin=jnp.max(agesspin),
                    )
                elif not cfg.complex and not cfg.sampling_only:
                    logging.info(f'Energy training iter {i:05d}: Loss={loss[0]:03.6f}, variance={auxdata.variance[0]:03.4f}, pmovepos={pmovepos[0]:0.2f}, pmovespin={pmovespin[0]:0.3f}, mcmc_width={mcmc_width[0]:0.2f}, mcmc_spin_width={mcmc_spin_width[0]:0.3f}, max_age_pos={jnp.max(agespos)}, max_age_spin={jnp.max(agesspin)}'
                                 )
                    writer.write(
                        i,
                        step=i,
                        loss=loss[0],
                        error=np.sqrt(
                            auxdata.variance[0] / (local_batch_size - 1)),
                        sigma=np.sqrt(auxdata.variance[0]),
                        variance=auxdata.variance[0],
                        pmovepos=pmovepos[0],
                        pmovespin=pmovespin[0],
                        mcmc_width=mcmc_width[0],
                        mcmc_spin_width=mcmc_spin_width[0],
                        max_age_pos=jnp.max(agespos),
                        max_age_spin=jnp.max(agesspin))

            if i % cfg.log.save_frequency == 0 and cfg.optim.optimizer != 'none':
                checkpoint.save_params(ckpt_save_path, params, i)
                stats.pop('loss')

                checkpoint.save(ckpt_save_path, i, pos, feats, opt_state,
                                mcmc_width, mcmc_spin_width, pmovepos, stats)

            if i % cfg.log.save_frequency == 0 and cfg.optim.optimizer == 'none':
                stats.pop('loss')

                loss, auxdata = constants.pmap(eval_loss)(params, (pos, feats))
                checkpoint.save(ckpt_save_path, i, pos, feats, opt_state, mcmc_width, mcmc_spin_width, pmovepos, {
                                'aux': auxdata, 'agespos': agespos, 'agesspin': agesspin})
                if cfg.sampling_only:
                    writer.write(
                        i,
                        step=i,
                        loss=loss[0],
                        error=np.sqrt(
                            auxdata.variance[0] / (local_batch_size - 1)),
                        sigma=np.sqrt(auxdata.variance[0]),
                        variance=auxdata.variance[0],
                        imaginary=auxdata.imaginary[0],
                        kinetic_re=jnp.mean(auxdata.kinetic.real),
                        kinetic_im=jnp.mean(auxdata.kinetic.imag),
                        potential=jnp.mean(auxdata.potential),
                        pmovepos=pmovepos[0],
                        pmovespin=pmovespin[0],
                        mcmc_width=mcmc_width[0],
                        mcmc_spin_width=mcmc_spin_width[0],
                        max_age_pos=jnp.max(agespos), max_age_spin=jnp.max(agesspin))
                    logging.info(
                        f'Energy training iter {i:05d}: Loss={loss[0]:03.6f}, variance={auxdata.variance[0]:03.4f}, pmovepos={pmovepos[0]:0.2f}, pmovespin={np.mean(pmovespin):0.3f}, mcmc_width={mcmc_width[0]:0.2f}, mcmc_spin_width={mcmc_spin_width[0]:0.3f}, max_age_pos={jnp.max(agespos)}, max_age_spin={jnp.max(agesspin)}')

            if i > 0 and i % cfg.mcmc.adapt_frequency == 0:
                if np.mean(pmovepos) > 0.55:
                    mcmc_width = np.maximum(
                        1.1*mcmc_width, 0.02*jnp.ones_like(mcmc_width))

                if np.mean(pmovepos) < 0.5:
                    mcmc_width = np.maximum(
                        mcmc_width/1.1, 0.02*jnp.ones_like(mcmc_width))

                if cfg.mcmc.spin_sample:
                    if np.mean(pmovespin) > 0.55:
                        mcmc_spin_width = np.maximum(
                            1.1*mcmc_spin_width, 0.002*jnp.ones_like(mcmc_spin_width))
                        # do not allow spin width greater than 0.95
                        mcmc_spin_width = np.minimum(
                            mcmc_spin_width, 0.95*jnp.ones_like(mcmc_spin_width))
                    if np.mean(pmovespin) < 0.5:
                        mcmc_spin_width = np.maximum(
                            mcmc_spin_width/1.1, 0.002*jnp.ones_like(mcmc_spin_width))
