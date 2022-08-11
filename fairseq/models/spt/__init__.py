# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .spt_config import (
    SPTConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .spt_decoder import SPTDecoder, SPTDecoderBase, Linear
from .spt_encoder import SPTEncoder, SPTEncoderBase
from .spt_legacy import (
    SPTModel,
    base_architecture,
    tiny_architecture,
    spt_iwslt_de_en,
    spt_wmt_en_de,
    spt_vaswani_wmt_en_de_big,
    spt_vaswani_wmt_en_fr_big,
    spt_wmt_en_de_big,
    spt_wmt_en_de_big_t2t,
)
from .spt_base import SPTModelBase, Embedding


__all__ = [
    "SPTModelBase",
    "SPTConfig",
    "SPTDecoder",
    "SPTDecoderBase",
    "SPTEncoder",
    "SPTEncoderBase",
    "SPTModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    "spt_iwslt_de_en",
    "spt_wmt_en_de",
    "spt_vaswani_wmt_en_de_big",
    "spt_vaswani_wmt_en_fr_big",
    "spt_wmt_en_de_big",
    "spt_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
