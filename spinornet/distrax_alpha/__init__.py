# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Distrax: Probability distributions in JAX."""

# Bijectors.
from spinornet.distrax_alpha._src.bijectors.bijector import Bijector
from spinornet.distrax_alpha._src.bijectors.bijector import BijectorLike
from spinornet.distrax_alpha._src.bijectors.block import Block
from spinornet.distrax_alpha._src.bijectors.chain import Chain
from spinornet.distrax_alpha._src.bijectors.gumbel_cdf import GumbelCDF
from spinornet.distrax_alpha._src.bijectors.inverse import Inverse
from spinornet.distrax_alpha._src.bijectors.lambda_bijector import Lambda
from spinornet.distrax_alpha._src.bijectors.lower_upper_triangular_affine import LowerUpperTriangularAffine
from spinornet.distrax_alpha._src.bijectors.masked_coupling import MaskedCoupling
from spinornet.distrax_alpha._src.bijectors.rational_quadratic_spline import RationalQuadraticSpline
from spinornet.distrax_alpha._src.bijectors.scalar_affine import ScalarAffine
from spinornet.distrax_alpha._src.bijectors.sigmoid import Sigmoid
from spinornet.distrax_alpha._src.bijectors.split_coupling import SplitCoupling
from spinornet.distrax_alpha._src.bijectors.tanh import Tanh
from spinornet.distrax_alpha._src.bijectors.unconstrained_affine import UnconstrainedAffine

# Distributions.
from spinornet.distrax_alpha._src.distributions.bernoulli import Bernoulli
from spinornet.distrax_alpha._src.distributions.categorical import Categorical
from spinornet.distrax_alpha._src.distributions.deterministic import Deterministic
from spinornet.distrax_alpha._src.distributions.distribution import Distribution
from spinornet.distrax_alpha._src.distributions.distribution import DistributionLike
from spinornet.distrax_alpha._src.distributions.epsilon_greedy import EpsilonGreedy
from spinornet.distrax_alpha._src.distributions.gamma import Gamma
from spinornet.distrax_alpha._src.distributions.greedy import Greedy
from spinornet.distrax_alpha._src.distributions.gumbel import Gumbel
from spinornet.distrax_alpha._src.distributions.independent import Independent
from spinornet.distrax_alpha._src.distributions.laplace import Laplace
from spinornet.distrax_alpha._src.distributions.log_stddev_normal import LogStddevNormal
from spinornet.distrax_alpha._src.distributions.logistic import Logistic
from spinornet.distrax_alpha._src.distributions.mixture_same_family import MixtureSameFamily
from spinornet.distrax_alpha._src.distributions.multinomial import Multinomial
from spinornet.distrax_alpha._src.distributions.mvn_diag import MultivariateNormalDiag
from spinornet.distrax_alpha._src.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from spinornet.distrax_alpha._src.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from spinornet.distrax_alpha._src.distributions.mvn_tri import MultivariateNormalTri
from spinornet.distrax_alpha._src.distributions.normal import Normal
from spinornet.distrax_alpha._src.distributions.one_hot_categorical import OneHotCategorical
from spinornet.distrax_alpha._src.distributions.quantized import Quantized
from spinornet.distrax_alpha._src.distributions.softmax import Softmax
from spinornet.distrax_alpha._src.distributions.straight_through import straight_through_wrapper
from spinornet.distrax_alpha._src.distributions.transformed import Transformed
from spinornet.distrax_alpha._src.distributions.uniform import Uniform

# Utilities.
from spinornet.distrax_alpha._src.utils.conversion import as_bijector
from spinornet.distrax_alpha._src.utils.conversion import as_distribution
from spinornet.distrax_alpha._src.utils.conversion import to_tfp
from spinornet.distrax_alpha._src.utils.hmm import HMM
from spinornet.distrax_alpha._src.utils.importance_sampling import importance_sampling_ratios
from spinornet.distrax_alpha._src.utils.math import multiply_no_nan
from spinornet.distrax_alpha._src.utils.monte_carlo import estimate_kl_best_effort
from spinornet.distrax_alpha._src.utils.monte_carlo import mc_estimate_kl
from spinornet.distrax_alpha._src.utils.monte_carlo import mc_estimate_kl_with_reparameterized
from spinornet.distrax_alpha._src.utils.monte_carlo import mc_estimate_mode
from spinornet.distrax_alpha._src.utils.transformations import register_inverse

__version__ = "0.1.2"

__all__ = (
    "as_bijector",
    "as_distribution",
    "straight_through_wrapper",
    "Bernoulli",
    "Bijector",
    "BijectorLike",
    "Block",
    "Categorical",
    "Chain",
    "Distribution",
    "DistributionLike",
    "EpsilonGreedy",
    "estimate_kl_best_effort",
    "Gamma",
    "Greedy",
    "Gumbel",
    "GumbelCDF",
    "HMM",
    "importance_sampling_ratios",
    "Independent",
    "Inverse",
    "Lambda",
    "Laplace",
    "LogStddevNormal",
    "Logistic",
    "LowerUpperTriangularAffine",
    "MaskedCoupling",
    "mc_estimate_kl",
    "mc_estimate_kl_with_reparameterized",
    "mc_estimate_mode",
    "MixtureSameFamily",
    "Multinomial",
    "multiply_no_nan",
    "MultivariateNormalDiag",
    "MultivariateNormalDiagPlusLowRank",
    "MultivariateNormalFullCovariance",
    "MultivariateNormalTri",
    "Normal",
    "OneHotCategorical",
    "Quantized",
    "RationalQuadraticSpline",
    "register_inverse",
    "ScalarAffine",
    "Sigmoid",
    "Softmax",
    "SplitCoupling",
    "to_tfp",
    "Transformed",
    "UnconstrainedAffine",
    "Uniform",
)


#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Distrax public API. /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
