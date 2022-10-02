# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.spt import (
    SPTConfig,
    SPTDecoderBase,
    SPTEncoderBase,
)
# For SPT
from fairseq.criterions.spt import _parsing


class SPTModelBase(FairseqEncoderDecoderModel):
    """
    SPT model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (SPTEncoder): the encoder
        decoder (SPTDecoder): the decoder

    The SPT model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.spt_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True
        self.phase = None

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, SPTConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # Build Embeddings
        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        # Build Encoder, Decoder
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return SPTEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return SPTDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if getattr(self, 'phase', None) == "pruning":
            compute_c = True
        else:
            compute_c = False

        _dev = self.encoder.embedding_c.data.device

        enc_pos_emb_mask = self.pruning_manager.get_embedding_mask("encoder", _dev=_dev)
        dec_pos_emb_mask = self.pruning_manager.get_embedding_mask("decoder", _dev=_dev)
        
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
            compute_c=compute_c, pos_emb_mask=enc_pos_emb_mask,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            compute_c=compute_c,
            pos_emb_mask=dec_pos_emb_mask,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


    '''
    @torch.no_grad()
    def pruning(self, gl_dict, eps=1e-8):
        en_heads = self.cfg.encoder.attention_heads
        de_heads = self.cfg.decoder.attention_heads
        
        for _n, _p in self.named_parameters():
            if '_c' in _n:
                set_param(self, _n, nn.Parameter(_p.data))
            elif 'embed_tokens' in _n:
                set_param(self, _n, nn.Parameter(_p.data))
                if 'decoder.embed_tokens' in _n:
                    self.decoder.output_projection.weight = self.decoder.embed_tokens.weight
                continue                
            elif 'output_projection' in _n:
                continue
            elif 'layer_norm' in _n or 'alpha' in _n:
                # Global pruning (TBD)
                set_param(self, _n, nn.Parameter(_p.data))
                continue
            else:
                ende, ly, type, wb = _parsing(_n)
                num_heads = en_heads if ende=='encoder' else de_heads
                if 'proj' in type:
                    attn_type = type.split('.')[0]
                    # Get _key
                    if 'q_proj' in type or 'k_proj' in type:
                        # qk proj
                        _key = f'{ende}.{ly}.{attn_type}.qk'
                    else:
                        # vo proj
                        _key = f'{ende}.{ly}.{attn_type}.vo'
                    _gl, _count = gl_dict[_key]
                    _mask = ((_gl/_count)>eps).repeat(num_heads)
                    # Perform pruning
                    if 'out_proj' in type:
                        if 'weight' in wb:
                            set_param(self, _n,
                                      nn.Parameter(_p.data[:, _mask]))
                        else:
                            # bias
                            set_param(self, _n,
                                      nn.Parameter(_p.data))
                            continue
                    else:
                        # q,k,v_proj
                        if 'weight' in wb:
                            """
                            ############### For Test ##################
                            if torch.sum(_mask) == 0:
                                print('\n\n+++++++++++++++++++++++++')
                                print(_p.data[0:10, 0:10])
                                print('++++++++++++++++++++++++++++')
                            ###########################################
                            """
                            set_param(self, _n,
                                      nn.Parameter(_p.data[_mask, :]))
                        else:
                            set_param(self, _n,
                                      nn.Parameter(_p.data[_mask]))
                elif 'fc' in type:
                    _key = f'{ende}.{ly}.fc'
                    _gl, _count = gl_dict[_key]
                    _mask = (_gl/_count)>eps
                    if 'fc1' in type:
                        if 'weight' in wb:
                            """
                            # adjust dropout rate for fc1
                            _ll = _n.split('.')[:-2]+\
                                    ['activation_dropout_module_fc1']
                            _dropout_name = '.'.join(_ll)
                            _dropout = recursive_get_param(self, _dropout_name)
                            _out, _in = _p.data.shape
                            new_out = torch.sum(_mask).item()
                            _dropout.p = _dropout.p * np.sqrt(new_out/ _out)
                            # adjust dropout rate for fc1 end
                            """
                            set_param(self, _n,
                                      nn.Parameter(_p.data[_mask, :]))
                        else:
                            set_param(self, _n,
                                      nn.Parameter(_p.data[_mask]))
                    else:
                        # fc2
                        if 'weight' in wb:
                            """
                            # adjust dropout rate for fc2
                            _ll = _n.split('.')[:-2]+\
                                    ['dropout_module_fc2']
                            _dropout_name = '.'.join(_ll)
                            _dropout = recursive_get_param(self, _dropout_name)
                            _out, _in = _p.data.shape
                            new_in = torch.sum(_mask).item()
                            _dropout.p = _dropout.p * np.sqrt(new_in/ _in)
                            # adjust dropout rate for fc1 end
                            """

                            set_param(self, _n,
                                      nn.Parameter(_p.data[:, _mask]))
                        else:
                            set_param(self, _n,
                                      nn.Parameter(_p.data))
                            continue
    '''
    @torch.no_grad()
    def decrease_c(self, _ratio=None):
        pm = self.pruning_manager
        pd = pm.pruning_dict

        ratio = _ratio if _ratio else pm.c_shrink_rate
        dec = pm._decreasing 
        
        # _N = 'encoder.layers.0.fc_c'
        """
        _N = 'decoder.layers.5.encoder_attn_vo_c'
        print("+++"*25)
        print("** Ratio: ", ratio)
        print("** Before shrinking")
        print(recursive_get_param(self, _N))
        """

        for _n in pd.keys():
            _indices = pd[_n]
            _p = recursive_get_param(self, _n)
            if dec[1] == 'a':
                # Arithmetic
                _p[_indices] = _p[_indices] - ratio
            elif dec[1] == 'g':
                # Geometric
                _p[_indices] = _p[_indices] * ratio
            else:
                raise Exception('Not an identified decreasing type')
        """
        print("** After shrinking")
        print(recursive_get_param(self, _N))
        """

    @torch.no_grad()
    def pruning(self,):
        pm = self.pruning_manager
        pd = pm.pruning_dict

        en_heads = self.cfg.encoder.attention_heads
        de_heads = self.cfg.decoder.attention_heads

        def get_pruning_mask(max_len, pruning_indices):
            _mask = torch.ones(max_len).bool()
            _mask[pruning_indices] = False
            return _mask

        for _n, _p in self.named_parameters():
            if _n[-2:] == "_c" :
                _indices = pd[_n] if _n in pd else []
                mask = get_pruning_mask(_p.shape, _indices) # its name is its key
                set_param(self, _n, nn.Parameter(_p.data[mask]))
                continue  
            elif 'embed_tokens' in _n:
                ende = _n.split('.')[0]
                _key = f"{ende}.embedding_c"
                mask = get_pruning_mask(_p.shape[1], pd[_key])
                set_param(self, _n, nn.Parameter(_p.data[:, mask]))
                if 'decoder.embed_tokens' in _n:
                    self.decoder.output_projection.weight = self.decoder.embed_tokens.weight

            elif 'output_projection' in _n:
                continue
 
            elif 'layer_norm' in _n:
                ende, ly, type, wb = _parsing(_n)
                _key = f"{ende}.embedding_c"
                _indices = pd[_key] if _key in pd else []
                mask = get_pruning_mask(_p.shape[0], _indices)
                set_param(self, _n, nn.Parameter(_p.data[mask]))

            elif 'fc' in _n:
                # fc layers
                # fc1: (gl_dim, fc_dim) | bias: fc_dim | global: prev_sub
                # fc2: (fc_dim, gl_dim) | bias: fc_dim | global: prev_sub
                ende, ly, type, wb = _parsing(_n)

                # Get global and local masks
                global_key = f'{ende}.embedding_c'
                local_key = f'{ende}.layers.{ly}.fc_c'

                global_indices = pd[global_key] if global_key in pd else []
                local_indices = pd[local_key] if local_key in pd else []

                if 'fc2' in _n:
                    if 'bias' in _n:
                        global_mask = get_pruning_mask(_p.shape[0],  global_indices)
                        set_param(self, _n, nn.Parameter(_p.data[global_mask]))
                    else:
                        global_mask = get_pruning_mask(_p.shape[0],  global_indices)
                        local_mask = get_pruning_mask(_p.shape[1],  local_indices)
                        new_p = _p.data[global_mask, :][:, local_mask]
                        set_param(self, _n, nn.Parameter(new_p.data))
                else:
                    if 'bias' in _n:
                        local_mask = get_pruning_mask(_p.shape[0],  local_indices)
                        set_param(self, _n, nn.Parameter(_p.data[local_mask]))
                    else:
                        global_mask = get_pruning_mask(_p.shape[1],  global_indices)
                        local_mask = get_pruning_mask(_p.shape[0],  local_indices)
                        new_p = _p.data[local_mask, :][:, global_mask]
                        set_param(self, _n, nn.Parameter(new_p.data))
            else:
                # qkvo_proj
                # q: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # k: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # v: (vo_dim, gl_dim) | bias: vo_dim | global: 
                # o: (gl_dim, vo_dim) | bias: gl_dim | global: previous sub-layer ln_c
                
                ende, ly, type, wb = _parsing(_n)
                # Get global and local masks
                if 'self_attn' in _n:
                    global_key = f'{ende}.embedding_c'
                    if 'q_proj' in _n or 'k_proj' in _n:
                        local_key = f'{ende}.layers.{ly}.self_attn_qk_c'
                    else:
                        local_key = f'{ende}.layers.{ly}.self_attn_vo_c'
                else:
                    # encoder_attn
                    if 'k_proj' in _n or 'v_proj' in _n:
                        global_key = f'encoder.embedding_c'
                    else:
                        global_key = f'decoder.embedding_c'
                        
                    if 'q_proj' in _n or 'k_proj' in _n:
                        local_key = f'{ende}.layers.{ly}.encoder_attn_qk_c'
                    else:
                        local_key = f'{ende}.layers.{ly}.encoder_attn_vo_c'

                # q: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # k: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # v: (vo_dim, gl_dim) | bias: vo_dim | global: 
                global_indices = pd[global_key] if global_key in pd else []
                local_indices = pd[local_key] if local_key in pd else []

                if 'out_proj' in _n:
                    if 'bias' in _n:
                        global_mask = get_pruning_mask(_p.shape[0],  global_indices)
                        set_param(self, _n, nn.Parameter(_p.data[global_mask]))
                    else:
                        global_mask = get_pruning_mask(_p.shape[0],  global_indices)
                        local_mask = get_pruning_mask(_p.shape[1],  local_indices)
                        new_p = _p.data[global_mask, :][:, local_mask]
                        set_param(self, _n, nn.Parameter(new_p.data))
                else:
                    if 'bias' in _n:
                        local_mask = get_pruning_mask(_p.shape[0],  local_indices)
                        set_param(self, _n, nn.Parameter(_p.data[local_mask]))
                    else:
                        global_mask = get_pruning_mask(_p.shape[1],  global_indices)
                        local_mask = get_pruning_mask(_p.shape[0],  local_indices)
                        new_p = _p.data[local_mask, :][:, global_mask]
                        set_param(self, _n, nn.Parameter(new_p.data))
    
    def get_num_groups(self,):
        param_dict = self.state_dict()
        num_groups = []
        for ende in ['encoder', 'decoder']:
            # Embeddings
            num_groups.append(param_dict[f'{ende}.embedding_c'].shape[0])
            for ly in range(0,6):
                # self_attn
                num_groups.append(param_dict[f'{ende}.layers.{ly}.self_attn_qk_c'].shape[0])
                num_groups.append(param_dict[f'{ende}.layers.{ly}.self_attn_vo_c'].shape[0])

                if ende == 'decoder':
                    # encoder-attn
                    num_groups.append(param_dict[f'{ende}.layers.{ly}.encoder_attn_qk_c'].shape[0])
                    num_groups.append(param_dict[f'{ende}.layers.{ly}.encoder_attn_vo_c'].shape[0]) 
                num_groups.append(param_dict[f'{ende}.layers.{ly}.fc_c'].shape[0])
        return num_groups
        

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def set_param(_model, _name, new_param):
    _attrs = _name.split('.')
    _parent = _model
    for _attr in _attrs[:-1]:
        _parent = getattr(_parent, _attr)
    setattr(_parent, _attrs[-1], new_param)

def recursive_get_param(_model, _name):
    _attrs = _name.split('.')
    _parent = _model
    for _attr in _attrs[:-1]:
        _parent = getattr(_parent, _attr)
    return getattr(_parent, _attrs[-1])
