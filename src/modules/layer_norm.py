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

import numbers
from typing import Tuple
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch import Tensor
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
        super().__init__()
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
        """Reset the parameters of the layer"""
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, _input: Tensor, embedding_c = None, weighted=False) -> Tensor:
        """Forward pass of the layer"""
        assert len(self.normalized_shape) == 1
        if weighted:
            if embedding_c is not None:
                batch_size, _layer, emb_dim = _input.shape
                _mu = (torch.sum(_input * embedding_c, dim=-1) \
                    / torch.sum(embedding_c)).view(batch_size, _layer, 1)
                _sum = torch.sum(((_input-_mu) ** 2) * embedding_c, dim=-1)
                sigma = torch.sqrt(_sum / torch.sum(embedding_c)).view(batch_size, _layer, 1)
            else:
                batch_size, _layer, emb_dim = _input.shape
                _mu = (torch.sum(_input, dim=-1) / emb_dim).view(batch_size, _layer, 1)
                _sum = torch.sum(((_input-_mu) ** 2), dim=-1)
                sigma = torch.sqrt(_sum / emb_dim).view(batch_size, _layer, 1)

        else:
            if embedding_c is not None:
                _input *= embedding_c
            batch_size, _layer, emb_dim = _input.shape
            _mu = (torch.sum(_input, dim = -1) / self.numel).view(batch_size, _layer, 1)
            _sum = torch.sum((_input - _mu) ** 2, dim=-1) + (_mu**2).squeeze(2) \
                * (self.numel-emb_dim)
            sigma = (torch.sqrt(_sum / self.numel)).view(batch_size, _layer, 1)


        norm_emb = (_input - _mu) / (sigma + self.eps)
        return norm_emb * self.weight + self.bias

    def extra_repr(self, ) -> str:
        return '{normalized_shape}, eps={eps}, '\
                'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


def layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    """LayerNorm"""
    return CustomLayerNorm(normalized_shape, eps, elementwise_affine)

class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm in fp32"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_):
        """Forward pass of the Fp32LayerNorm"""
        output = F.layer_norm(
            input_.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input_)
