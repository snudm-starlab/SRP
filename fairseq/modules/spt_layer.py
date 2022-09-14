# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.models.spt import (
    SPTConfig,
)


class SPTEncoderLayerBase(nn.Module):
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
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.dropout_module_fc2 = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.activation_dropout_module_fc1 = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
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

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        #### Connection parameters
        # These parameters are only used in 
        if hasattr(self.cfg, "c_init"):
            c_init = self.cfg.c_init
        else:
            c_init = 1.
        
        self.self_attn_qk_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        self.self_attn_vo_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        self.self_attn_ln_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)

        self.fc_c = torch.nn.Parameter(torch.ones(cfg.encoder.ffn_embed_dim) * c_init)
        self.fc_ln_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        
        # Connection parameters

    def reset_c(self,):
        _dev = self.fc1.weight.device
        c_init = self.cfg.c_init
        self.self_attn_qk_c.data.fill_(c_init) #  = torch.ones_like(self.self_attn_qk_c).to(_dev) * c_init
        self.self_attn_vo_c.data.fill_(c_init) # = torch.ones_like(self.self_attn_vo_c).to(_dev) * c_init
        self.self_attn_ln_c.data.fill_(c_init) # = torch.ones_like(self.self_attn_ln_c).to(_dev) * c_init

        self.fc_c.data.fill_(c_init) # = torch.ones_like(self.fc_c.data).to(_dev) * c_init
        self.fc_ln_c.data.fill_(c_init) #  = torch.ones_like(self).to(_dev) * c_init

        

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def _get_fc_rank(self, remove_num: int) -> List[int]:
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
    """
    def _prune_fc_layer(self, remove_index: List[int]):
        new_fc1_weight = []
        new_fc1_bias = []
        for i in range(self.fc1.out_features):
            if i not in remove_index:
                new_fc1_weight.append(self.fc1.weight[i])
                new_fc1_bias.append(self.fc1.bias[i])

        new_fc1_weight = torch.stack(new_fc1_weight).detach()
        new_fc1_weight.requires_grad = True

        new_fc1_bias = torch.stack(new_fc1_bias).detach()
        new_fc1_bias.requires_grad = True

        self.fc1 = quant_noise(
            nn.Linear(self.fc1.in_features, self.fc1.out_features - len(remove_index)),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc1.weight = torch.nn.Parameter(new_fc1_weight)
        self.fc1.bias = torch.nn.Parameter(new_fc1_bias)

        new_fc2_weight = []
        new_fc2_bias = []
        for i in range(self.fc2.in_features):
            if i not in remove_index:
                new_fc2_weight.append(self.fc2.weight[:, i])
        new_fc2_bias = self.fc2.bias.detach()

        new_fc2_weight = torch.stack(new_fc2_weight, dim=-1).detach()
        new_fc2_weight.requires_grad = True

        new_fc2_bias = self.fc2.bias.detach()
        new_fc2_bias.requires_grad = True

        self.fc2 = quant_noise(
            nn.Linear(self.fc2.in_features - len(remove_index), self.fc2.out_features),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc2.weight = torch.nn.Parameter(new_fc2_weight)
        self.fc2.bias = torch.nn.Parameter(new_fc2_bias)
    """

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        scoring=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
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
        
        # Self-attention starts
        if scoring:
            self.reset_c()

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
                    attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
                )
            residual = x
            if scoring:
                qk_c = self.self_attn_qk_c 
                vo_c = self.self_attn_vo_c
            else:
                qk_c, vo_c = None, None

            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                qk_c=qk_c,
                vo_c=vo_c,
            )

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            x = self.self_attn_layer_norm(x)
            if scoring:
                x *= self.self_attn_ln_c

        # Self-attention end
        if self.fc1.weight.shape[0] == 0:
            # Skip fc layers
            fc_result = x
        else:
            residual = x
            x = self.activation_fn(self.fc1(x))
            if scoring:
                x *= self.fc_c
            x = self.activation_dropout_module_fc1(x)
            x = self.fc2(x)

            fc_result = x

            x = self.dropout_module_fc2(x)
            x = self.residual_connection(x, residual)
            x = self.final_layer_norm(x)
            if scoring:
                x *= self.fc_ln_c

        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x


# backward compatible with the legacy argparse format
class SPTEncoderLayer(SPTEncoderLayerBase):
    def __init__(self, args):
        super().__init__(SPTConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, SPTConfig.from_namespace(args)
        )


class SPTDecoderLayerBase(nn.Module):
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

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_ln = (
            LayerNorm(self.embed_dim)
            if utils.safe_getattr(cfg, "scale_attn", False)
            else None
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.nh,)), requires_grad=True)
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
        self.activation_dropout_module_fc1 = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        """
        self.ffn_layernorm = (
            LayerNorm(cfg.decoder.ffn_embed_dim)
            if utils.safe_getattr(cfg, "scale_fc", False)
            else None
        )
        
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if utils.safe_getattr(cfg, "scale_resids", False)
            else None
        )
        """
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

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False

        # These parameters are only used in 
        # c_init = cfg.c_init
        if hasattr(self.cfg, "c_init"):
            c_init = self.cfg.c_init
        else:
            c_init = 1.
        
        self.self_attn_qk_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        self.self_attn_vo_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        self.self_attn_ln_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)

        self.encoder_attn_qk_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        self.encoder_attn_vo_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        self.encoder_attn_ln_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)

        self.fc_c = torch.nn.Parameter(torch.ones(cfg.encoder.ffn_embed_dim) * c_init)
        self.fc_ln_c = torch.nn.Parameter(torch.ones(self.embed_dim) * c_init)
        
        # Connection parameters

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
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

    def reset_c(self, ):
        _dev = self.fc1.weight.device
        c_init = self.cfg.c_init
        self.self_attn_qk_c.data.fill_(c_init) # = torch.ones(self.embed_dim).to(_dev) * c_init
        self.self_attn_vo_c.data.fill_(c_init) #  = torch.ones(self.embed_dim).to(_dev) * c_init
        self.self_attn_ln_c.data.fill_(c_init) #  = torch.ones(self.embed_dim).to(_dev) * c_init

        self.encoder_attn_qk_c.data.fill_(c_init) #  = torch.ones(self.embed_dim).to(_dev) * c_init
        self.encoder_attn_vo_c.data.fill_(c_init) #  = torch.ones(self.embed_dim).to(_dev) * c_init
        self.encoder_attn_ln_c.data.fill_(c_init) #  = torch.ones(self.embed_dim).to(_dev) * c_init

        self.fc_c.data.fill_(c_init) #  = torch.ones(self.cfg.encoder.ffn_embed_dim).to(_dev) * c_init
        self.fc_ln_c.data.fill_(c_init) #  = torch.ones(self.embed_dim).to(_dev) * c_init

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        scoring=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
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
        
        if scoring:
            self_attn_qk_c = self.self_attn_qk_c
            self_attn_vo_c = self.self_attn_vo_c
            encoder_attn_qk_c = self.encoder_attn_qk_c 
            encoder_attn_vo_c = self.encoder_attn_vo_c
            self.reset_c()
        else:
            self_attn_qk_c = None
            self_attn_vo_c = None
            encoder_attn_qk_c = None 
            encoder_attn_vo_c = None


        # Masked self attention starts
        if self.self_attn.v_proj.weight.shape[0] == 0:
            # Skip masked self attention
            attn = None
            pass
        else:
            residual = x
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
                        (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                    )
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
                y = torch.cat((encoder_out, x), dim=0)
            else:
                y = x

            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                qk_c=self_attn_qk_c,
                vo_c=self_attn_vo_c,
                )
            if self.c_attn is not None:
                tgt_len, bsz = x.size(0), x.size(1)
                x = x.view(tgt_len, bsz, self.nh, self.head_dim)
                x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
                x = x.reshape(tgt_len, bsz, self.embed_dim)
            if self.attn_ln is not None:
                x = self.attn_ln(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            x = self.self_attn_layer_norm(x)
            if scoring:
                x *= self.self_attn_ln_c
        
        # Masked self attention end

        # Encoder-attention starts
        
        if self.encoder_attn is not None and encoder_out is not None:
            if self.encoder_attn.v_proj.weight.shape[0] == 0: 
                pass
            else:
                residual = x
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

                x, attn = self.encoder_attn(
                    query=x,
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
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
                x = self.encoder_attn_layer_norm(x)
                if scoring:
                    x *= self.encoder_attn_ln_c
        # Encoder attention end

        if self.fc1.weight.shape[0] == 0:
            # Skip fc layers is they are pruned
            pass
        else: 
            residual = x
            x = self.activation_fn(self.fc1(x))
            if scoring:
                x *= self.fc_c
            x = self.activation_dropout_module_fc1(x)

            # if self.ffn_layernorm is not None:
            #     x = self.ffn_layernorm(x)
            x = self.fc2(x)
            x = self.dropout_module_fc2(x)

            # if self.w_resid is not None:
            #     residual = torch.mul(self.w_resid, residual)
            x = self.residual_connection(x, residual)
            x = self.final_layer_norm(x)
            if scoring:
                x *= self.fc_ln_c
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
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class SPTDecoderLayer(SPTDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            SPTConfig.from_namespace(args),
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
            SPTConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            SPTConfig.from_namespace(args),
        )
