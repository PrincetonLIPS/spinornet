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


def default() -> ml_collections.ConfigDict:
    cfg = ml_collections.ConfigDict({
        'batch_size': 4096,
        'use_x64': True,
        'sampling_only': False,
        'complex': False,
        'debug': {
            'deterministic': False,
        },
        'pretrain': {
            'lr': 3e-4,
            'iterations': 1000,
            'optimizer': 'kfac',  # one of 'adam', 'kfac'.
            # pretrain not recommended without basis set HF.
            'hf_method': 'HFOrbital',
        },
        'log': {
            'restore_path': '',
            'restore_epoch': 0,
            'save_path': '',
            'save_frequency': 1000
        },
        'system': {
            'pyscf_cell': None,
            'type': None
        },
        'init_spin_type': 'random',  # can be random, half, or nelec
        'soc_alpha': 0.0,
        'mcmc': {
            'burn_in': 100,
            'steps': 10,
            'init_config_width': 0.0005,
            'init_width': 0.5,  # can make this 0.0 to fix e- positions.
            'init_spin_width': 0.3,
            'spin_sample': True,
            'adapt_frequency': 100,
            'pos_type': 'langevin',
            'spin_type': 'cont'  # can be cont or discrete
        },
        'network': {
            'ndets': 1,
            'network_type': 'spinornet',
            'hf_method': 'HFOrbital',
            'twist': (0.0, 0.0, 0.0),
            'ferminet': {'hidden_size': 32,
                         'hidden_size_single': 256},
            'spinornet': {'hidden_size': 32,
                          'hidden_size_single': 256},
            'spinornet_single': {
                'hidden_size_single': 256},
            'spinornet_pbc': {'hidden_size': 32,
                              'hidden_size_single': 256, 'feattype': 'cas'},
            'spinornet_pbc2dHEG': {'hidden_size': 32,
                                   'hidden_size_single': 256,
                                   'feattype': 'cas',
                                   'inc_ae': True, 'inc_rae': False},

        },

        'optim': {
            'optimizer': 'kfac',  # one of 'adam', 'kfac', 'spring','none' .
            'iterations': 1001,  # number of iterations
            'ministeps': 1,
            'lr': {
                'rate': 0.05,  # learning rate
                'decay': 1.0,  # exponent of learning rate decay
                'delay': 10000.0,  # term that sets the scale of the rate decay
                'percent': 0.001,  # amount to adjust learn rate for variance min
            },
            'kfac': {
                'invert_every': 1,
                'cov_update_every': 1,
                'damping': 0.001,
                'cov_ema_decay': 0.95,
                'momentum': 0.0,
                'momentum_type': 'regular',
                # Warning: adaptive damping is not currently available.
                'min_damping': 1.0e-4,
                'norm_constraint': 0.001,
                'mean_center': True,
                'l2_reg': 0.0,
                'register_only_generic': False,
            },
            # If greater than zero, scale (at which to clip local energy) in units
            # of the mean deviation from the mean.
            'clip_local_energy': 5.0,
            'center_at_clip': True,
            'lap_method': 'scan',  # scan or vmap or folx.
            'lap_use_vmap': False,  # flag used for rashba calc only.
            'max_vmap_batch_size': 0,
            # ADAM hyperparameters. See optax documentation for details.
            # Adam not extensively tested.
            'adam': {
                'b1': 0.9,
                'b2': 0.999,
                'eps': 1.0e-8,
                'eps_root': 0.0,
            },
            "spring": {
                # SPRING hyperparams
                "mu": 0.99,
                "momentum": 0.0,  # non-zero value not recommended
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
            },
        },


    })
    return cfg
