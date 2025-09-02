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

from spinornet import base_config
from pyscf.gto import Mole
import numpy as np


def get_config(input_str):
    cfg = base_config.default()
    L = float(input_str)
    n = 3
    angles = np.zeros((n, 3))
    angles[:, 0] = np.cos(np.arange(n) * 2*np.pi/n)
    angles[:, 1] = np.sin(np.arange(n) * 2*np.pi/n)
    atom_pos = angles * L / np.sqrt(2-2*np.cos(2*np.pi/n))
    mol = Mole()
    mol.atom = [("H", pos) for pos in atom_pos]
    mol.basis = 'ccpvdz'
    mol.spin = 1
    mol.unit = 'B'
    mol.build()
    cfg.system.pyscf_cell = mol
    cfg.pbc = 0
    cfg.system.type = 'mol'
    cfg.complex = True
    cfg.network.hf_method = 'HFOrbital'
    cfg.is2d = False
    return cfg
