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
import numpy as np
from spinornet.utils import system


def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
    """Returns square lattice vectors with Wigner-Seitz radius rs."""
    volume = np.pi * (rs**2) * nelec
    length = volume**(1 / 2)
    return length * np.eye(3)


def get_config(input_str):
    rs, nelec1, nelec2, socalpha = input_str.split(',')
    rs = float(rs)
    socalpha = float(socalpha)
    nelec = (int(nelec1), int(nelec2))
    cfg = base_config.default()

    lattice = _sc_lattice_vecs(rs, sum(nelec))
    kpoints = system.make_kpoints2d(lattice, nelec, sum(nelec)*2)
    cfg.pbc = 1
    cfg.system.type = 'heg2d'
    cfg.complex = True
    cfg.is2d = True

    simulation_cell = system.HEGCell({
        'nelec': nelec,
        'a': lattice,
        'original_cell': system.HEGCell({'a': lattice}),
        'klist': kpoints,
    })
    cfg.system.pyscf_cell = simulation_cell
    cfg.soc_alpha = socalpha
    return cfg
