# Copyright 2025 Spinornet authors.

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

import jax
import os
import itertools
from absl import app
from ml_collections.config_flags import config_flags
from absl import flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Path to config file.')

def main(_):
  from spinornet import process
  from spinornet.configs.heg2d import get_config
  from spinornet import base_config
  cfg = base_config.default()
  cfg.use_x64=False
  if cfg.use_x64:
    jax.config.update("jax_enable_x64", True)

  cfg.update(get_config('5,9,1,0.1'))
  cfg.batch_size = 2048
  cfg.optim.iterations = 30001
  cfg.log.save_frequency = 1000
  cfg.optim.optimizer = 'kfac'
  cfg.pretrain.iterations = 0
  cfg.mcmc.init_width=1
  cfg.mcmc.adapt_frequency=10
  cfg.mcmc.pos_type='langevin'
  cfg.mcmc.spin_type='cont'
  cfg.init_spin_type = 'nelec'
  cfg.mcmc.steps=10
  cfg.mcmc.burn_in=100
  cfg.optim.lap_method='folx'
  cfg.optim.max_vmap_batch_size=0

  cfg.network.network_type = 'spinornet_pbc2dHEG'
  cfg.debug.deterministic=False

  units = [2,3,4,5]

  for i, wftype in itertools.product(range(4),
                                            ['noncoll', 'pnoncoll']):
        if wftype == 'noncoll':
            cfg.network.hf_method = 'HFOrbitalHEG2dRb'
            cfg.mcmc.spin_sample = True
        elif wftype == 'pnoncoll':
            cfg.network.hf_method = 'HFOrbitalHEG2dRb'
            cfg.mcmc.spin_sample = False

        cfg.network.spinornet_pbc2dHEG.hidden_size = units[i]
        cfg.network.spinornet_pbc2dHEG.hidden_size_single = units[i]

        cfg.log.save_path = f'heg2d_10_lb01_spinord_{wftype}_{units[i]}_{units[i]}_2l'

        cfg.optim.lr.rate = 0.1
        cfg.optim.lr.decay = 1
        cfg.optim.lr.delay = 10000
        cfg.optim.kfac.damping = 0.001

        #cfg.log.restore_path = f'3_data_code/2deg/{wftype}-{units[i]}/'
        #cfg.log.restore_epoch = -1

        process.process(cfg)


if __name__ == '__main__':
    app.run(main)