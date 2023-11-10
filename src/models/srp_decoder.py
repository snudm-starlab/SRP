"""
Starlab Transformer Compression with SRP (Selectively Regularized Pruning)

Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
        U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.0
Date : Nov 29, 2022
Main Contact: Hyojin Jeon

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
This code is mainly based on the [GitHub Repository]
[GitHub Repository]: https://github.com/facebookresearch/fairseq
"""

import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .srp_config import SRPConfig
from ..modules import srp_layer, layer_norm


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    """Return module name for backward compatibility in `make_generation_fast_`"""
    if module_name == "SRPDecoderBase":
        return "SRPDecoder"
    else:
        return module_name


class SRPDecoderBase(FairseqIncrementalDecoder):
    """
    SRP decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`SRPDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            linear_layer(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        self.embedding_c = nn.Parameter(torch.ones(embed_dim), requires_grad=True)
        self.pos_emb_mask = nn.Parameter(torch.zeros(0), requires_grad=False)

        if cfg.layernorm_embedding:
            self.layernorm_embedding = layer_norm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = layer_norm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            linear_layer(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        """Build the output projection layer."""
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        """Build a single decoder layer."""
        layer = srp_layer.SRPDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        compute_c=False,
        use_kd=False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if use_kd:
            _input, extra, hidden_states = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                full_context_alignment=full_context_alignment,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                compute_c=compute_c,
                use_kd=use_kd,
            )
        else:
            _input, extra = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                full_context_alignment=full_context_alignment,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                compute_c=compute_c,
                use_kd=use_kd,
            )

        if not features_only:
            if compute_c:
                _input*=self.embedding_c
            _input = self.output_layer(_input)

        if use_kd:
            return _input, extra, hidden_states
        return _input, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        compute_c: bool = False,
        use_kd=False,
    ):
        """return decoder features only without applying output layer."""
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            compute_c=compute_c,
            use_kd=use_kd,
        )


    # A scriptable subclass of this class has an extract_features method and calls
    # super().extract_features, but super() is not supported in torchscript. A copy of
    # this function is made to be used in the subclass instead.


    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        compute_c=False,
        use_kd=False,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with SRP Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        _input = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            _input = self.quant_noise(_input)

        if self.project_in_dim is not None:
            _input = self.project_in_dim(_input)

        if positions is not None:
            _input += positions[:, :, self.pos_emb_mask]

        if self.layernorm_embedding is not None:
            _input = self.layernorm_embedding(_input)

        _input = self.dropout_module(_input)

        if compute_c:
            _input = _input * self.embedding_c


        # B x T x C -> T x B x C
        _input = _input.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        hidden_states = {}
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [_input]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(_input)
            else:
                self_attn_mask = None
            if not use_kd:
                _input, layer_attn, _ = layer(
                    _input,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    embedding_c=self.embedding_c,
                    compute_c=compute_c,
                )
            else:
                _input, layer_attn, _, _hs = layer(
                    _input,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    embedding_c=self.embedding_c,
                    compute_c=compute_c,
                    use_kd=use_kd,
                )
                hidden_states[f'decoder.self_attn.{idx}'] = _hs['self_attn']
                hidden_states[f'decoder.encoder_attn.{idx}'] = _hs['encoder_attn']
                hidden_states[f'decoder.fc.{idx}'] = _hs['fc']

            inner_states.append(_input)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(_input)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            _input = self.layer_norm(_input)

        # T x B x C -> B x T x C
        _input = _input.transpose(0, 1)

        if self.project_out_dim is not None:
            _input = self.project_out_dim(_input)
        if use_kd:
            return _input, {"attn": [attn], "inner_states": inner_states}, hidden_states

        return _input, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        """Cached future mask."""
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript.
        # This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = f"{name}.embed_positions.weights"
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                f"{name}.embed_positions._float_tensor"
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for _wb in ("weight", "bias"):
                    k = f"{name}.layers.{i}.layer_norms.{old}.{_wb}"
                    if k in state_dict:
                        state_dict[
                            f"{name}.layers.{i}.{new}.{_wb}"
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = f"{name}.version"
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def linear_layer(in_features, out_features, bias=True):
    """Linear layer"""
    _module = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(_module.weight)
    if bias:
        nn.init.constant_(_module.bias, 0.0)
    return _module


class SRPDecoder(SRPDecoderBase):
    """SRP decoder."""
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            SRPConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        """Build the output projection layer."""
        super().build_output_projection(
            SRPConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        """Build a single decoder layer."""
        return super().build_decoder_layer(
            SRPConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )
