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

from absl import app
from ml_collections.config_flags import config_flags
from absl import flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config', None, 'Path to config file.')

def main(_):
  cfg = FLAGS.config
  if cfg.use_x64:
    jax.config.update("jax_enable_x64", True)
  from spinornet import process
  process.process(cfg)

if __name__ == '__main__':
    app.run(main)