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

import spinornet.kfac_alpha as kfac_alpha
from typing import Mapping, Optional, Sequence, Any, Dict
import chex
import jax
import numpy as np
from spinornet.kfac_alpha._src import curvature_blocks
import jax.numpy as jnp
from spinornet.kfac_alpha._src import utils
from math import prod

Array = kfac_alpha.utils.Array
Numeric = kfac_alpha.utils.Numeric

# name, #num_inputs, #num_outputs
extra_dim_dense_tag = kfac_alpha.LayerTag("extra_dim_dense_tag", 1, 1)


def register_extra_dim_dense(y, x, w, b):
    if b is None:
        return extra_dim_dense_tag.bind(y, x, w)
    return extra_dim_dense_tag.bind(y, x, w, b)


def _extra_dim_dense(x, params):
    w, *opt_b = params
    y = x @ w
    return y if not opt_b else y + opt_b[0]


def _extra_dim_dense_parameter_extractor(
    eqns,
):
    """Extracts all parameters from the conv_general_dilated operator."""
    for eqn in eqns:
        if eqn.primitive.name == "dot_general":
            return dict(**eqn.params)
    assert False


extra_dim_dense_with_bias_pattern = kfac_alpha.tag_graph_matcher.GraphPattern(
    name="extra_dim_dense",
    tag_primitive=extra_dim_dense_tag,
    compute_func=_extra_dim_dense,
    parameters_extractor_func=_extra_dim_dense_parameter_extractor,
    example_args=[np.zeros([100, 2, 3]), [np.zeros([3, 50]), np.zeros([50])]],
)

extra_dim_dense_with_bias_pattern2 = kfac_alpha.tag_graph_matcher.GraphPattern(
    name="extra_dim_dense2",
    tag_primitive=extra_dim_dense_tag,
    compute_func=_extra_dim_dense,
    parameters_extractor_func=_extra_dim_dense_parameter_extractor,
    example_args=[np.zeros([100, 52, 4, 2, 3]), [
        np.zeros([3, 32]), np.zeros([32])]],
)

extra_dim_dense_no_bias_pattern2 = kfac_alpha.tag_graph_matcher.GraphPattern(
    name="extra_dim_dense",
    tag_primitive=extra_dim_dense_tag,
    compute_func=_extra_dim_dense,
    parameters_extractor_func=_extra_dim_dense_parameter_extractor,
    example_args=[np.zeros([100, 52, 4, 2, 3]), [np.zeros([3, 32]), ]],
)

extra_dim_dense_no_bias_pattern = kfac_alpha.tag_graph_matcher.GraphPattern(
    name="extra_dim_dense",
    tag_primitive=extra_dim_dense_tag,
    compute_func=_extra_dim_dense,
    parameters_extractor_func=_extra_dim_dense_parameter_extractor,
    example_args=[np.zeros([100, 2, 3]), [np.zeros([3, 50]),]],
)


GRAPH_PATTERNS = (extra_dim_dense_with_bias_pattern,
                  extra_dim_dense_no_bias_pattern,
                  extra_dim_dense_with_bias_pattern2,
                  extra_dim_dense_no_bias_pattern2,
                  ) + kfac_alpha.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS


class ExtraDimDenseTwoKroneckerFactored(kfac_alpha.TwoKroneckerFactored):
    """A :class:`~TwoKroneckerFactored` block specifically for dense layers."""

    # @utils.auto_scope_method
    def _update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        batch_size,
    ):
        # Copy this first since we mutate it later in this function.
        state = state.copy()

        [x] = estimation_data["inputs"]
        [dy] = estimation_data["outputs_tangent"]

        assert utils.first_dim_is_size(batch_size, x, dy)
        if self.has_bias:
            x_one = jnp.ones_like(x[..., :1])
            x = jnp.concatenate([x, x_one], axis=-1)
        batch_size = prod(x.shape[:-1])
        input_stats = jnp.einsum("...y,...z->yz", x, x) / batch_size
        output_stats = jnp.einsum("...y,...z->yz", dy, dy) / batch_size
        state.factors[0].update(input_stats, ema_old, ema_new)
        state.factors[1].update(output_stats, ema_old, ema_new)

        return state


kfac_alpha.set_default_tag_to_block_ctor(
    "extra_dim_dense_tag", ExtraDimDenseTwoKroneckerFactored)
