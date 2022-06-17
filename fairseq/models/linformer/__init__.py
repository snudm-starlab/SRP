# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .linformer_config import (
    LinformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .linformer_decoder import LinformerDecoder, LinformerDecoderBase, Linear
from .linformer_encoder import LinformerEncoder, LinformerEncoderBase
from .linformer_legacy import (
    LinformerModel,
    base_architecture,
    tiny_architecture,
    linformer_iwslt_de_en,
    linformer_wmt_en_de,
    linformer_vaswani_wmt_en_de_big,
    linformer_vaswani_wmt_en_fr_big,
    linformer_wmt_en_de_big,
    linformer_wmt_en_de_big_t2t,
)
from .linformer_base import LinformerModelBase, Embedding


__all__ = [
    "LinformerModelBase",
    "LinformerConfig",
    "LinformerDecoder",
    "LinformerDecoderBase",
    "LinformerEncoder",
    "LinformerEncoderBase",
    "LinformerModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    "linformer_iwslt_de_en",
    "linformer_wmt_en_de",
    "linformer_vaswani_wmt_en_de_big",
    "linformer_vaswani_wmt_en_fr_big",
    "linformer_wmt_en_de_big",
    "linformer_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
