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

from typing import Dict, List, Optional

import torch
from torch import nn
from torch import Tensor

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from . import MultiheadAttention, layer_norm
from ..models.srp_config import SRPConfig


class SRPEncoderLayerBase(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, return_fc=False):
        super().__init__()
        self.cfg = cfg
        self.return_fc = return_fc
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = layer_norm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        self.do_weighted = cfg.weighted_layernorm


        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = layer_norm(self.embed_dim, export=cfg.export)

        self.self_attn_qk_c = torch.nn.Parameter(torch.ones(self.embed_dim))
        self.self_attn_vo_c = torch.nn.Parameter(torch.ones(self.embed_dim))

        self.fc_c = torch.nn.Parameter(torch.ones(cfg.encoder.ffn_embed_dim))

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        """Build the first layer of the feed-forward block."""
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        """Build the second layer of the feed-forward block."""
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def _get_fc_rank(self, remove_num: int) -> List[int]:
        """Get the indices of the filters to be removed in the FC layer"""
        f1_filter_param = []
        for i in range(self.fc1.out_features):
            f1_filter_param.append(
                torch.sum(torch.abs(self.fc1.weight[i]))
                + torch.sum(torch.abs(self.fc2.weight[:, i]))
                + torch.abs(self.fc1.bias[i])
            )
        return sorted(
            range(len(f1_filter_param)), key=lambda k: f1_filter_param[k], reverse=False
        )[0:remove_num]

    def build_self_attention(self, embed_dim, cfg):
        """Build self-attention layer."""
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, _x, residual):
        """Residual connection"""
        return residual + _x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for wb_type in ("weight", "bias"):
                k = f"{name}.layer_norms.{old}.{wb_type}"
                if k in state_dict:
                    state_dict[f"{name}.{new}.{wb_type}"] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        _input,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        compute_c = False,
        embedding_c = None,
        use_kd = False
    ):
        """
        Args:
            _input (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        hidden_states = {}

        if self.self_attn.v_proj.weight.shape[0] == 0:
            # Skip self-attention
            pass
        else:
            # anything in original attn_mask = 1, becomes -1e8
            # anything in original attn_mask = 0, becomes 0
            # Note that we cannot use -inf here, because at some edge cases,
            # the attention weight (before softmax) for some padded element in query
            # will become -inf, which results in NaN in model parameters
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(
                    attn_mask.to(torch.bool), -1e8 if _input.dtype == torch.float32 else -1e4
                )
            residual = _input
            if compute_c:
                qk_c = self.self_attn_qk_c
                vo_c = self.self_attn_vo_c
            else:
                qk_c, vo_c = None, None

            if use_kd:
                _input, _, self_att = self.self_attn(
                    query=_input,
                    key=_input,
                    value=_input,
                    key_padding_mask=encoder_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                    qk_c=qk_c,
                    vo_c=vo_c,
                    return_A=True
                )
                hidden_states['self_attn'] = self_att

            else:
                _input, _ = self.self_attn(
                    query=_input,
                    key=_input,
                    value=_input,
                    key_padding_mask=encoder_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                    qk_c=qk_c,
                    vo_c=vo_c,
                )

            _input = self.dropout_module(_input)

            if compute_c:
                _input = self.residual_connection(_input, residual)
                _input = self.self_attn_layer_norm(_input, embedding_c=embedding_c,
                                             weighted = self.do_weighted)
                _input *= embedding_c
            else:
                _input = self.residual_connection(_input, residual)
                _input = self.self_attn_layer_norm(_input, weighted = self.do_weighted)

        if self.fc1.weight.shape[0] == 0:
            # Skip fc layers
            fc_result = _input
        else:
            residual = _input
            _input = self.activation_fn(self.fc1(_input))
            _input = self.activation_dropout_module(_input)
            if compute_c:
                _input = _input * self.fc_c
            _input = self.fc2(_input)
            hidden_states['fc'] = _input

            fc_result = _input

            _input = self.dropout_module(_input)
            if compute_c:
                _input = self.residual_connection(_input, residual)
                _input = self.final_layer_norm(_input, embedding_c=embedding_c,
                                          weighted = self.do_weighted)
                _input*= embedding_c
            else:
                _input = self.residual_connection(_input, residual)
                _input = self.final_layer_norm(_input, weighted = self.do_weighted)

        if self.return_fc and not torch.jit.is_scripting():
            return _input, fc_result
        if use_kd:
            return _input, hidden_states
        return _input


# backward compatible with the legacy argparse format
class SRPEncoderLayer(SRPEncoderLayerBase):
    """Encoder layer block."""
    def __init__(self, args):
        super().__init__(SRPConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, SRPConfig.from_namespace(args)
        )


class SRPDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.dropout_module_fc2 = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention
        self.do_weighted = cfg.weighted_layernorm

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_ln = (
            layer_norm(self.embed_dim)
            if utils.safe_getattr(cfg, "scale_attn", False)
            else None
        )
        self.num_heads = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.num_heads,)), requires_grad=True)
            if scale_heads
            else None
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.activation_dropout_module_fc1 = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

        self.self_attn_layer_norm = layer_norm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = layer_norm(self.embed_dim, export=cfg.export)


        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = layer_norm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False
        self.self_attn_qk_c = torch.nn.Parameter(torch.ones(self.embed_dim))
        self.self_attn_vo_c = torch.nn.Parameter(torch.ones(self.embed_dim))
        self.encoder_attn_qk_c = torch.nn.Parameter(torch.ones(self.embed_dim))
        self.encoder_attn_vo_c = torch.nn.Parameter(torch.ones(self.embed_dim))
        self.fc_c = torch.nn.Parameter(torch.ones(cfg.encoder.ffn_embed_dim))

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        """Build the first layer of the feed-forward block."""
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        """Build the second layer of the feed-forward block."""
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        """Build self-attention layer."""
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        """Build encoder attention layer."""
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        """Prepare model for ONNX export."""
        self.onnx_trace = True

    def residual_connection(self, _x, residual):
        """Residual connection"""
        return residual + _x

    def forward(
        self,
        _input,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        embedding_c=None,
        compute_c=False,
        use_kd = False,
    ):
        """
        Args:
            _input (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        if compute_c:
            self_attn_qk_c = self.self_attn_qk_c
            self_attn_vo_c = self.self_attn_vo_c
            encoder_attn_qk_c = self.encoder_attn_qk_c
            encoder_attn_vo_c = self.encoder_attn_vo_c
        else:
            self_attn_qk_c = None
            self_attn_vo_c = None
            encoder_attn_qk_c = None
            encoder_attn_vo_c = None

        hidden_states = {}

        # Masked self-attention start
        if self.self_attn.v_proj.weight.shape[0] == 0:
            # Skip masked self attention
            attn = None
            pass
        else:
            residual = _input
            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state, saved_state)
            _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
            if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
            ):
                if self_attn_mask is not None:
                    assert encoder_out is not None
                    self_attn_mask = torch.cat(
                        (_input.new_zeros(_input.size(0), encoder_out.size(0)), self_attn_mask)
                        , dim=1)
                if self_attn_padding_mask is not None:
                    if encoder_padding_mask is None:
                        assert encoder_out is not None
                        encoder_padding_mask = self_attn_padding_mask.new_zeros(
                            encoder_out.size(1), encoder_out.size(0)
                        )
                    self_attn_padding_mask = torch.cat(
                        (encoder_padding_mask, self_attn_padding_mask), dim=1
                    )
                assert encoder_out is not None
                _y = torch.cat((encoder_out, _input), dim=0)
            else:
                _y = _input
            if use_kd:
                _input, attn, _as = self.self_attn(
                    query=_input,
                    key=_y,
                    value=_y,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,
                    need_weights=False,
                    attn_mask=self_attn_mask,
                    qk_c=self_attn_qk_c,
                    vo_c=self_attn_vo_c,
                    return_A=True,
                    )
                hidden_states['self_attn'] = _as
            else:
                _input, attn = self.self_attn(
                    query=_input,
                    key=_y,
                    value=_y,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,
                    need_weights=False,
                    attn_mask=self_attn_mask,
                    qk_c=self_attn_qk_c,
                    vo_c=self_attn_vo_c,
                    )


            if self.c_attn is not None:
                tgt_len, bsz = _input.size(0), _input.size(1)
                _input = _input.view(tgt_len, bsz, self.num_heads, self.head_dim)
                _input = torch.einsum("tbhd,h->tbhd", _input, self.c_attn)
                _input = _input.reshape(tgt_len, bsz, self.embed_dim)

            if self.attn_ln is not None:
                _input = self.attn_ln(_input)

            _input = self.dropout_module(_input)
            if compute_c:
                _input = self.residual_connection(_input, residual)
                _input = self.self_attn_layer_norm(_input, embedding_c=embedding_c,
                                             weighted = self.do_weighted)
                _input *= embedding_c
            else:
                _input = self.residual_connection(_input, residual)
                _input = self.self_attn_layer_norm(_input, weighted = self.do_weighted)
        # Masked self attention end

        # Encoder-attention start
        if self.encoder_attn is not None and encoder_out is not None:
            if self.encoder_attn.v_proj.weight.shape[0] == 0:
                pass
            else:
                residual = _input
                if prev_attn_state is not None:
                    prev_key, prev_value = prev_attn_state[:2]
                    saved_state: Dict[str, Optional[Tensor]] = {
                        "prev_key": prev_key,
                        "prev_value": prev_value,
                    }
                    if len(prev_attn_state) >= 3:
                        saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                    assert incremental_state is not None
                    self.encoder_attn._set_input_buffer(incremental_state, saved_state)
                if use_kd:
                    _input, attn, _ae = self.encoder_attn(
                        query=_input,
                        key=encoder_out,
                        value=encoder_out,
                        key_padding_mask=encoder_padding_mask,
                        incremental_state=incremental_state,
                        static_kv=True,
                        need_weights=need_attn or (not self.training and self.need_attn),
                        need_head_weights=need_head_weights,
                        qk_c=encoder_attn_qk_c,
                        vo_c=encoder_attn_vo_c,
                        return_A=True,
                        )
                    hidden_states['encoder_attn'] = _ae
                else:
                    _input, attn = self.encoder_attn(
                        query=_input,
                        key=encoder_out,
                        value=encoder_out,
                        key_padding_mask=encoder_padding_mask,
                        incremental_state=incremental_state,
                        static_kv=True,
                        need_weights=need_attn or (not self.training and self.need_attn),
                        need_head_weights=need_head_weights,
                        qk_c=encoder_attn_qk_c,
                        vo_c=encoder_attn_vo_c,
                        )

                _input = self.dropout_module(_input)
                if compute_c:
                    _input = self.residual_connection(_input, residual)
                    _input = self.encoder_attn_layer_norm(_input, embedding_c=embedding_c,
                                                    weighted = self.do_weighted)
                    _input *= embedding_c
                else:
                    _input = self.residual_connection(_input, residual)
                    _input = self.encoder_attn_layer_norm(_input, weighted = self.do_weighted)
        # Encoder-attention ends

        # FNN starts
        if self.fc1.weight.shape[0] == 0:
            # Skip fc layers is they are pruned
            pass
        else:
            residual = _input
            _input = self.activation_fn(self.fc1(_input))
            _input = self.activation_dropout_module_fc1(_input)
            if compute_c:
                _input = _input * self.fc_c

            _input = self.fc2(_input)
            hidden_states['fc'] = _input
            _input = self.dropout_module_fc2(_input)
            if compute_c:
                _input = self.residual_connection(_input, residual)
                _input = self.final_layer_norm(_input, embedding_c=embedding_c,
                                          weighted = self.do_weighted)
                _input *= embedding_c
            else:
                _input = self.residual_connection(_input, residual)
                _input = self.final_layer_norm(_input, weighted = self.do_weighted)
        # FNN ends

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return _input, attn, self_attn_state
        if use_kd:
            return _input, attn, None, hidden_states
        return _input, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        """Prepare model for fast decoding."""
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class SRPDecoderLayer(SRPDecoderLayerBase):
    """Decoder layer block."""
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            SRPConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            SRPConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            SRPConfig.from_namespace(args),
        )
