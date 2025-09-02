# Copyright 2020 DeepMind Technologies Limited.
#
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

import attr
from spinornet.utils import elements
from spinornet.utils import units as unit_conversion
from typing import Sequence, Union, Tuple, Optional, Mapping, Any
import jax.numpy as jnp
import itertools
import numpy as np
from ml_collections import ConfigDict


@attr.s
class Atom:
    """Atom information for Hamiltonians.

    The nuclear charge is inferred from the symbol if not given, in which case the
    symbol must be the IUPAC symbol of the desired element.

    Attributes:
      symbol: Element symbol.
      coords: An iterable of atomic coordinates. Always a list of floats and in
        bohr after initialisation. Default: place atom at origin.
      charge: Nuclear charge. Default: nuclear charge (atomic number) of atom of
        the given name.
      atomic_number: Atomic number associated with element. Default: atomic number
        of element of the given symbol. Should match charge unless fractional
        nuclear charges are being used.
      units: String giving units of coords. Either bohr or angstrom. Default:
        bohr. If angstrom, coords are converted to be in bohr and units to the
        string 'bohr'.
      coords_angstrom: list of atomic coordinates in angstrom.
      coords_array: Numpy array of atomic coordinates in bohr.
      element: elements.Element corresponding to the symbol.
    """
    symbol = attr.ib(type=str)
    coords = attr.ib(
        type=Sequence[float],
        converter=lambda xs: tuple(float(x) for x in xs),
        default=(0.0, 0.0, 0.0))
    charge = attr.ib(type=float, converter=float)
    atomic_number = attr.ib(type=int, converter=int)
    units = attr.ib(
        type=str,
        default='bohr',
        validator=attr.validators.in_(['bohr', 'angstrom']))

    @charge.default
    def _set_default_charge(self):
        return self.element.atomic_number

    @atomic_number.default
    def _set_default_atomic_number(self):
        return self.element.atomic_number

    def __attrs_post_init__(self):
        if self.units == 'angstrom':
            self.coords = [unit_conversion.angstrom2bohr(
                x) for x in self.coords]
            self.units = 'bohr'

    @property
    def coords_angstrom(self):
        return [unit_conversion.bohr2angstrom(x) for x in self.coords]

    @property
    def coords_array(self):
        if not hasattr(self, '_coords_arr'):
            self._coords_arr = np.array(self.coords)
        return self._coords_arr

    @property
    def element(self):
        return elements.SYMBOLS[self.symbol]


class HEGCell(ConfigDict):
    def __init__(self, initial_dictionary: Optional[Mapping[str, Any]] = None, type_safe: bool = True, convert_dict: bool = True):
        super().__init__(initial_dictionary, type_safe, convert_dict)
        self.BV = np.linalg.inv(self.a.T)*2*np.pi
        self.AV = np.linalg.inv(self.BV).T

    def atom_coords(self):
        return np.array([[0., 0., 0.]])

    def atom_charges(self):
        return np.zeros(1, dtype=int)

    def lattice_vectors(self):
        return self.a


def make_kpoints(
    lattice: Union[np.ndarray, jnp.ndarray],
    spins: Tuple[int, int],
    min_kpoints: Optional[int] = None,
) -> jnp.ndarray:
    """Generates an array of reciprocal lattice vectors.

    Args:
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim). (Note that ndim=3 is currently
        a hard-coded default).
      spins: Tuple of the number of spin-up and spin-down electrons.
      min_kpoints: If specified, the number of kpoints which must be included in
        the output. The number of kpoints returned will be the
        first filled shell which is larger than this value. Defaults to None,
        which results in min_kpoints == sum(spins).

    Raises:
      ValueError: Fewer kpoints requested by min_kpoints than number of
        electrons in the system.

    Returns:
      jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
        vectors sorted in ascending order according to length.
    """
    rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice)
    # Calculate required no. of k points
    if min_kpoints is None:
        min_kpoints = sum(spins)
    elif min_kpoints < sum(spins):
        raise ValueError(
            'Number of kpoints must be equal or greater than number of electrons')

    dk = 1 + 1e-5
    # Generate ordinals of the lowest min_kpoints kpoints
    max_k = int(jnp.ceil(min_kpoints * dk)**(1 / 3.))
    ordinals = sorted(range(-max_k, max_k+1), key=abs)
    ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))

    kpoints = ordinals @ rec_lattice.T
    kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
    k_norms = jnp.linalg.norm(kpoints, axis=1)

    return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]


def make_kpoints2d(
    lattice,
    spins,
    min_kpoints=None,
) -> jnp.ndarray:
    """Generates an array of reciprocal lattice vectors.

    Args:
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim). (Note that ndim=2 is currently
        a hard-coded default).
      spins: Tuple of the number of spin-up and spin-down electrons.
      min_kpoints: If specified, the number of kpoints which must be included in
        the output. The number of kpoints returned will be the
        first filled shell which is larger than this value. Defaults to None,
        which results in min_kpoints == sum(spins).

    Raises:
      ValueError: Fewer kpoints requested by min_kpoints than number of
        electrons in the system.

    Returns:
      jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
        vectors sorted in ascending order according to length.
    """
    rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice[:2, :2])
    # Calculate required no. of k points
    if min_kpoints is None:
        min_kpoints = sum(spins)
    elif min_kpoints < sum(spins):
        raise ValueError(
            'Number of kpoints must be equal or greater than number of electrons')

    dk = 1 + 1e-5
    # Generate ordinals of the lowest min_kpoints kpoints
    max_k = int(jnp.ceil(min_kpoints * dk)**(1 / 2.))
    ordinals = sorted(range(-max_k, max_k+1), key=abs)
    ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=2)))

    kpoints = ordinals @ rec_lattice.T

    kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
    k_norms = jnp.linalg.norm(kpoints, axis=1)

    return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]
