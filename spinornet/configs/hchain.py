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

from pyscf.pbc import gto
from spinornet import base_config
from spinornet import supercell
import numpy as np


def get_config(input_str):
    symbol, Sx, Sy, Sz, L, spin, basis = input_str.split(',')
    Sx = int(Sx)
    Sy = int(Sy)
    Sz = int(Sz)
    S = np.diag([Sx, Sy, Sz])
    L = float(L)
    spin = int(spin)
    cfg = base_config.default()

    # Set up cell
    cell = gto.Cell()
    cell.atom = f"""
    {symbol} {0}   {0}   {0}
    {symbol} 0 0 {L}
    """
    cell.basis = basis
    cell.a = np.array([[100, 0,   0],
                       [0, 100, 0],
                       [0, 0, 2*L]])
    cell.unit = "B"
    cell.spin = spin
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    simulation_cell.hf_type = 'uhf'
    cfg.system.pyscf_cell = simulation_cell
    cfg.pbc = 1
    cfg.system.type = 'solid'
    cfg.complex = True
    cfg.is2d = False
    return cfg
