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

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor, Size
from typing import Union, List, Tuple
import numbers
import numpy as np

class CustomLayerNorm(nn.Module):
    """
    Weighted Layer Normalization
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    def __init__(self, normalized_shape, eps=1e-5, \
                elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.numel = np.product(self.normalized_shape)
        self.dim = len(self.normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    
    def forward(self, x: Tensor, embedding_c = None, weighted=False) -> Tensor:
        assert len(self.normalized_shape) == 1
        if weighted:
            if embedding_c is not None:
                bs, l, emb_dim = x.shape
                mu = (torch.sum(x * embedding_c, dim=-1) / torch.sum(embedding_c)).view(bs, l, 1)
                _sum = torch.sum(((x-mu) ** 2) * embedding_c, dim=-1)
                sigma = torch.sqrt(_sum / torch.sum(embedding_c)).view(bs, l, 1)
            else:
                bs, l, emb_dim = x.shape
                mu = (torch.sum(x, dim=-1) / emb_dim).view(bs, l, 1)
                _sum = torch.sum(((x-mu) ** 2), dim=-1)
                sigma = torch.sqrt(_sum / emb_dim).view(bs, l, 1)

        else:
            if embedding_c is not None:
                x *= embedding_c
            bs, l, emb_dim = x.shape
            mu = (torch.sum(x, dim = -1) / self.numel).view(bs, l, 1)
            _sum = torch.sum((x - mu) ** 2, dim=-1) + (mu**2).squeeze(2) * (self.numel-emb_dim) 
            sigma = (torch.sqrt(_sum / self.numel)).view(bs, l, 1)    

        
        norm_emb = (x - mu) / (sigma + self.eps)
        return norm_emb * self.weight + self.bias

    def extra_repr(self, ) -> str:
        return '{normalized_shape}, eps={eps}, '\
                'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
        

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return CustomLayerNorm(normalized_shape, eps, elementwise_affine)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

