################################################################################
# Starlab Transformer Compression with SRP (Selectively Regularized Pruning)
#
# Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 29, 2022
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/facebookresearch/fairseq
################################################################################

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel

from .srp_config import SRPConfig
from .srp_encoder import SRPEncoderBase
from .srp_decoder import SRPDecoderBase

class SRPModelBase(FairseqEncoderDecoderModel):
    """
    SRP model based on `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (SRPEncoder): the encoder
        decoder (SRPDecoder): the decoder

    The SRP model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.srp_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

        _src_words = encoder.embed_tokens.weight.shape[0]
        _tar_words = decoder.embed_tokens.weight.shape[0]
        self.pm = PruningManager(cfg, _src_words, _tar_words)
        self.update_pos_emb_mask() # initialize embedding_mask
        self.phase = 'warming-up'

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, SRPConfig(), delete_default=False, with_prefix=""
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
        return SRPEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return SRPDecoderBase(
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
        use_kd=False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if getattr(self, 'phase', None) == 'pruning':
            compute_c = True
        else:
            compute_c = False

        if not use_kd:
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
                compute_c=compute_c,
            )
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
                compute_c=compute_c
            )
            return decoder_out
        else:
            hidden_states = {}
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
                compute_c=compute_c, use_kd=use_kd
            )
            hidden_states.update(encoder_out["hidden_states"])
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
                compute_c=compute_c,
                use_kd=use_kd
            )
            hidden_states.update(decoder_out[-1])
            return decoder_out, hidden_states



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
    
    def update_pos_emb_mask(self,):
        _dev = self.encoder.embedding_c.data.device
        # Update indices
        self.pm.update_embedding_mask('encoder')
        self.pm.update_embedding_mask('decoder')

        # Update embedding masks
        self.encoder.pos_emb_mask.data = self.pm.encoder_indices.to(_dev)
        self.decoder.pos_emb_mask.data = self.pm.decoder_indices.to(_dev)
    
    @torch.no_grad()
    def decrease_c(self, _ratio=None):
        pm = self.pm
        pd = pm.pruning_dict

        ratio = _ratio if _ratio else pm.c_shrink_rate
        dec = pm._decreasing         

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

    @torch.no_grad()
    def pruning(self,):
        pm = self.pm
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
            elif 'mask' in _n:
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
            num_layer = self.cfg.encoder_layers if ende == 'encoder' else self.cfg.decoder_layers
            for ly in range(0, num_layer):
                # self_attn
                num_groups.append(param_dict[f'{ende}.layers.{ly}.self_attn_qk_c'].shape[0])
                num_groups.append(param_dict[f'{ende}.layers.{ly}.self_attn_vo_c'].shape[0])

                if ende == 'decoder':
                    # encoder-attn
                    num_groups.append(param_dict[f'{ende}.layers.{ly}.encoder_attn_qk_c'].shape[0])
                    num_groups.append(param_dict[f'{ende}.layers.{ly}.encoder_attn_vo_c'].shape[0]) 
                num_groups.append(param_dict[f'{ende}.layers.{ly}.fc_c'].shape[0])
        return num_groups
        
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


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class PruningManager():
    def __init__(self, cfg, src_words, tar_words): 
        self.src_words = src_words
        self.tar_words = tar_words
        
        self.en_layers = cfg.encoder_layers
        self.de_layers = cfg.decoder_layers

        self.en_heads = cfg.encoder_attention_heads
        self.de_heads = cfg.decoder_attention_heads

        self.GLE = cfg.encoder_embed_dim
        self.GLD = cfg.decoder_embed_dim
        self.FC = cfg.encoder_ffn_embed_dim * self.en_layers \
                + cfg.decoder_ffn_embed_dim * self.de_layers

        # Assume that emb_encdoer == emb_decoder
        self.QK = cfg.encoder_embed_dim // self.en_heads
        self.VO = cfg.encoder_embed_dim // self.de_heads

        self.count = 0
        self._count = 0

        self.pruning_dict = {}

        # Compute pruing ratio    
        self.step_per_epoch = 1102 #TODO: step per epoch is hard-coded
        self.pruning_iter = cfg.pruning_iter
        self.pruning_period = cfg.pruning_period
        self.warming_up = cfg.warming_up
        self.compression_rate = cfg.compression_rate

        # self.P = 1 - np.sqrt(cfg.common.compression_rate)
        self.P = 1-self.get_pruning_rate(self.compression_rate)
        self.p = 1 - (1-self.P) ** (1/self.pruning_iter) 

        # For positional embedding
        self.encoder_orig_dim = cfg.encoder_embed_dim
        self.decoder_orig_dim = cfg.decoder_embed_dim

        self.encoder_indices = torch.arange(self.encoder_orig_dim)
        self.decoder_indices = torch.arange(self.decoder_orig_dim)
        
        self._decreasing = cfg.decreasing
        if self._decreasing == 'eg':
            self.c_shrink_rate = (1e-5) ** (1/ (cfg.pruning_period))
        elif self._decreasing == 'ea':
            self.c_shrink_rate =  1 / (cfg.pruning_period)
        elif self._decreasing == 'sg':
            self.c_shrink_rate = (1e-5) ** (1/ (cfg.pruning_period * self.step_per_epoch-1))
        elif self._decreasing == 'sa':
            self.c_shrink_rate =  (1 / (cfg.pruning_period * self.step_per_epoch-1))
        else:
            raise Exception('Not an identified decreasing type')

        # Step-wise shrink (ratio)


    def get_pruning_rate(self, compression_rate):
        assert self.GLE == self.GLD # For simplify computation of n1
        assert self.en_heads == self.de_heads

        _QK = self.GLE // self.en_heads * self.en_layers \
                + self.GLD // self.de_heads * self.de_layers * 2 
        _VO = self.GLE // self.en_heads * self.en_layers \
                + self.GLD // self.de_heads * self.de_layers * 2 

        n1 = float(self.GLE * (self.FC * 2 + _QK * self.en_heads * 2 + _VO * self.en_heads * 2))
        n2 = float(self.src_words * self.GLE + self.tar_words * self.GLD
            + self.GLE * 2 * (2*self.en_layers + 3*self.de_layers) # layer_norm
            + self.QK * self.en_heads * 2 
            + self.VO * self.en_heads + self.GLE * (self.en_layers + self.de_layers * 2)
            + self.FC + self.GLE * (self.en_layers + self.de_layers)
        )
        print("#### Original paramters: ", n1+n2) 
        A = (n2 / n1)
        B = -1 * compression_rate * (n1 + n2) / n1
        
        p = (-1 * A + np.sqrt(A ** 2 - 4 * B) ) / 2
        return p 

    def get(self, stage=0):
        if self.count > self.pruning_iter-1:
            # Do not pruning
            return -1, -1, -1, -1, -1
        gle = int(np.ceil(self.GLE * self.p))
        gld = int(np.ceil(self.GLD * self.p))
        fc = int(np.ceil(self.FC * self.p))
        qk = int(np.ceil(self.QK * self.p))
        vo = int(np.ceil(self.VO * self.p))
        
        if stage == 1:
            fc = 0; qk = 0 ; vo=0
            self.count += 0.5
        elif stage ==2:
            gle = 0 ; gld = 0
            self.count += 0.5
        else:
            self.count += 1      

        self.GLE -= gle
        self.GLD -= gld
        self.FC -= fc
        self.QK -= qk
        self.VO -= vo
         
        return gle, gld, fc, qk, vo

    def update_embedding_mask(self, ende):
        # ende: encoder or decoder
        GL = self.GLE if ende == 'encoder' else self.GLD
        if f'{ende}.embedding_c' in self.pruning_dict:
            emb_mask = self.pruning_dict[f'{ende}.embedding_c']
            emb_mask2 = torch.zeros(GL + emb_mask.shape[0]).bool()
            emb_mask2[emb_mask] = True
        else:
            emb_mask2 = torch.zeros(GL).bool()
        if ende == 'encoder':
            self.encoder_indices.data = self.encoder_indices[~emb_mask2]
        else:
            self.decoder_indices.data = self.decoder_indices[~emb_mask2]   

    def get_phase(self, e):
        if e <= self.warming_up:
            _p = 'warming-up'
        elif e <= self.warming_up + self.pruning_iter * self.pruning_period:
            _p = 'pruning'
            self._count = (e - self.warming_up) % self.pruning_period
        else:
            _p = 'fine-tuning'
            
        if self.pruning_period == 1:
            do_pruning = (e - self.warming_up) > 1 and \
                         (e - self.warming_up - self.pruning_iter) <= 1
        else:
            do_pruning = ((e - self.warming_up) > 1) and \
                        ((e - self.warming_up) % self.pruning_period == 0) \
                        and (e - self.warming_up) -  (self.pruning_period * self.pruning_iter) <= 1
        return _p, do_pruning

    def get_global_dict(self, model, gl, ende):
        gl_dict = {} # k:v = param_name: pruning indices
        _n = f"{ende}.embedding_c"
        score = self._get_attr(model, _n).grad
        gl_dict[_n] = torch.argsort(score, descending=True)[:gl]
        return gl_dict

    def get_qkvo_dict(self, model, pn, qkvo):
        pruning_dict = {} # k:v = param_name: pruning indices
        for ende in ["encoder", "decoder"]:
            module_list = ["self_attn"]
            if ende == "decoder":
                module_list.append("encoder_attn")
            n_layers = self.en_layers if ende=='encoder' else self.de_layers
            for ly in range(n_layers):
                for module in module_list:
                    _n = f"{ende}.layers.{ly}.{module}_{qkvo}_c"
                    _grad = self._get_attr(model, _n).grad
                    if _grad is None:
                        continue

                    _head_dim = _grad.shape[-1]//4
                    args_list = []
                    for i in range(4):
                        temp_grad, _args = torch.sort(_grad[_head_dim*i:_head_dim*(i+1)], descending=True)

                        _args += _head_dim * i
                        args_list.append(_args[:pn])
                    _inds_all = torch.cat(args_list)
                    pruning_dict[_n] = _inds_all

        return pruning_dict     

    def get_fc_dict(self, model, fc):
        fc_dict = {} # k:v = param_name: pruning indices
        scores = []
    
        for ende in ['encoder', 'decoder']:
            n_layers = self.en_layers if ende=='encoder' else self.de_layers
            for ly in range(0,n_layers):
                _n = f'{ende}.layers.{ly}.fc_c'
                score = self._get_attr(model, _n).grad
                if score is None:
                    continue
                scores.append(score)
        scores = torch.sort(torch.cat(scores), descending=True)[0]
        thres = scores[fc]
        count = 0
        for ende in ['encoder', 'decoder']:
            n_layers = self.en_layers if ende=='encoder' else self.de_layers
            for ly in range(0,n_layers):
                _n = f'{ende}.layers.{ly}.fc_c'
                score = self._get_attr(model, _n).grad
                if score is None:
                    continue
                cond = (score > thres)
                if torch.sum(cond) > 0:
                    count += torch.where(cond)[0].shape[0]
                    fc_dict[_n] = torch.where(cond)[0]
        return fc_dict


    def _get_attr(self, _model, _name):
        # Get attribute of the model
        _attrs = _name.split(".")
        _parent = _model
        for _attr in _attrs[:-1]:
            _parent = self._get_attr(_parent, _attr)
        return getattr(_parent, _attrs[-1])

def _parsing(_name):
    # Parse the name of parameters 
    assert 'embed_tokens' not in _name
    _l = _name.split('.')
    if 'attn' in _name and 'layer_norm' not in _name:
        ende, ly, type, wb = _l[0], _l[2], f'{_l[3]}.{_l[4]}',_l[5]
    else:
        try:
            ende, ly, type, wb = _l[0], _l[2], _l[3],_l[4]
        except Exception:
            print("* Name: ", _name)
    return ende, ly, type, wb
