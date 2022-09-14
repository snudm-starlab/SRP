# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class
from omegaconf import II, OmegaConf

# For SPT
from fairseq.criterions.spt import _parsing

logger = logging.getLogger(__name__)


@dataclass
class FairseqAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")


@register_optimizer("adam", dataclass=FairseqAdamConfig)
class FairseqAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqAdamConfig, params):
        super().__init__(cfg)
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(cfg, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if getattr(cfg, "tpu", False):
            if self.cfg.fp16_adam_stats:
                raise NotImplementedError("--fp16-adam-stats is only supported on GPU")
            # on TPUs we use the Adam defined here, since it
            # automatically casts gradients to FP32
            self._optimizer = Adam(params, **self.optimizer_config)
        elif use_fused_adam:
            logger.info("using FusedAdam")
            self._optimizer = fused_adam_cls(
                params, use_fp16_stats=self.cfg.fp16_adam_stats, **self.optimizer_config
            )
        else:
            if self.cfg.fp16_adam_stats:
                raise NotImplementedError(
                    "--fp16-adam-stats is only supported with FusedAdamV1"
                )
            self._optimizer = Adam(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(Adam, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss

    def pruning(self, _model):
        pm = _model.pruning_manager
        pd = pm.pruning_dict

        en_heads = _model.cfg.encoder.attention_heads
        de_heads = _model.cfg.decoder.attention_heads

        named_params = list(_model.named_parameters())
        # param_list = list(_model.parameters())
        param_list = []
        param_names = []
        for _n, _p in _model.named_parameters():
            if _n[-2:] == "_c":
                continue
            param_list.append(_p)
            param_names.append(_n)
            

        # model_params = list(param_list)
        self.param_groups[0]['params'] = param_list

        # named_params = list(_model.named_parameters())
        _dict = {}

        def get_pruning_mask(max_len, pruning_indices):
            _mask = torch.ones(max_len).bool()
            _mask[pruning_indices] = False
            return _mask
        """
        print("================= *********** ++++++++++++++++++++++++++++")
        print("Len of state.items()", len(self.state.items()))
        print("Len of param list", len(param_list))
        _count = 0
        for _n, _p in _model.named_parameters():
            if "_c" in _n:
                _count +=1
        print("_C count: ", _count)
        print("================= *********** ++++++++++++++++++++++++++++")
        """


        _i = 0
        for _k, _v in self.state.items():
            # _n = named_params[_i][0]
            # while _n[-2:] == '_c':
            #     _i +=1
            #     _n = named_params[_i][0]
            _n = param_names[_i]
            _shape = _v['exp_avg'].shape
            # print("*** ", _n, ": ", _shape)
            if _n[-2:] == "_c" :
                """
                _indices = pd[_n] if _n in pd else []
                mask = get_pruning_mask(_shape, _indices) # its name is its key
                _v['exp_avg'] = torch.zeros_like(mask)
                _v['exp_avg_sq'] = torch.zeros_like(mask)

                # set_param(self, _n, nn.Parameter(_p.data[mask]))
                """
                continue

            elif 'alpha' in _n: 
                ende = _n.split('.')[0]
                _key = f"{ende}.embedding_c"
                mask = get_pruning_mask(_shape[0], pd[_key])
                _v['exp_avg'] = _v['exp_avg'][mask]
                _v['exp_avg_sq'] = _v['exp_avg_sq'][mask]
                

            elif 'embed_tokens' in _n:
                ende = _n.split('.')[0]
                _key = f"{ende}.embedding_c"
                mask = get_pruning_mask(_shape[1], pd[_key])
                _v['exp_avg'] = _v['exp_avg'][:, mask]
                _v['exp_avg_sq'] = _v['exp_avg_sq'][:, mask]
                # if 'decoder.embed_tokens' in _n:
                #     self.decoder.output_projection.weight = self.decoder.embed_tokens.weight
            elif 'output_projection' in _n:
                continue

            elif 'layer_norm' in _n:
                ende, ly, type, wb = _parsing(_n)
                if 'self' in type:
                    _type = 'self_attn'
                elif 'encoder' in type:
                    _type = 'encoder_attn'
                else:
                    _type = 'fc'
                _key = f"{ende}.layers.{ly}.{_type}_ln_c"
                mask = get_pruning_mask(_shape[0], pd[_key])

                _v['exp_avg'] = _v['exp_avg'][mask]
                _v['exp_avg_sq'] = _v['exp_avg_sq'][mask]

            elif 'fc' in _n:
                # fc layers
                # fc1: (gl_dim, fc_dim) | bias: fc_dim | global: prev_sub
                # fc2: (fc_dim, gl_dim) | bias: fc_dim | global: prev_sub
                ende, ly, type, wb = _parsing(_n)

                # Get global and local masks
                if ende == 'encoder':
                    global_key = f'{ende}.layers.{ly}.self_attn_ln_c'
                else:
                    # decoder
                    global_key = f'{ende}.layers.{ly}.encoder_attn_ln_c'
                local_key = f'{ende}.layers.{ly}.fc_c'

                global_indices = pd[global_key] if global_key in pd else []
                local_indices = pd[local_key] if local_key in pd else []

                if 'fc2' in _n:
                    if 'bias' in _n:
                        global_mask = get_pruning_mask(_shape[0],  global_indices)
                        # set_param(self, _n, nn.Parameter(_p.data[global_mask]))
                        _v['exp_avg'] = _v['exp_avg'][global_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][global_mask]
                    else:
                        global_mask = get_pruning_mask(_shape[0],  global_indices)
                        local_mask = get_pruning_mask(_shape[1],  local_indices)
                        # new_p = _p.data[global_mask, :][:, local_mask]
                        # set_param(self, _n, nn.Parameter(new_p.data))
                        _v['exp_avg'] = _v['exp_avg'][global_mask,:][:,local_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][global_mask,:][:,local_mask]
                else:
                    if 'bias' in _n:
                        local_mask = get_pruning_mask(_shape[0],  local_indices)
                        # set_param(self, _n, nn.Parameter(_p.data[local_mask]))
                        _v['exp_avg'] = _v['exp_avg'][local_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][local_mask]
                    else:
                        global_mask = get_pruning_mask(_shape[1],  global_indices)
                        local_mask = get_pruning_mask(_shape[0],  local_indices)
                        # new_p = _p.data[local_mask, :][:, global_mask]
                        # set_param(self, _n, nn.Parameter(new_p.data))
                        _v['exp_avg'] = _v['exp_avg'][local_mask,:][:,global_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][local_mask,:][:,global_mask]
            else:
                # qkvo_proj
                # q: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # k: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # v: (vo_dim, gl_dim) | bias: vo_dim | global: 
                # o: (gl_dim, vo_dim) | bias: gl_dim | global: previous sub-layer ln_c
                
                ende, ly, type, wb = _parsing(_n)
                # Get global and local masks
                if 'self_attn' in _n:
                    if ly == '0':
                        global_key = f'{ende}.embedding_c'
                    else:
                        global_key = f'{ende}.layers.{int(ly)-1}.fc_ln_c'
                    if 'q_proj' in _n or 'k_proj' in _n:
                        local_key = f'{ende}.layers.{ly}.self_attn_qk_c'
                    else:
                        local_key = f'{ende}.layers.{ly}.self_attn_vo_c'
                else:
                    # encoder_attn
                    global_key = f'{ende}.layers.{ly}.self_attn_ln_c'
                    if 'q_proj' in _n or 'k_proj' in _n:
                        local_key = f'{ende}.layers.{ly}.encoder_attn_qk_c'
                    else:
                        local_key = f'{ende}.layers.{ly}.encoder_attn_vo_c'

                # q: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # k: (qk_dim, gl_dim) | bias: qk_dim | global: 
                # v: (vo_dim, gl_dim) | bias: vo_dim | global: 

                global_indices = pd[global_key] if global_key in pd else []
                local_indices = pd[local_key] if local_key in pd else []

                # Compute loss 
                if 'out_proj' in _n:
                    if 'bias' in _n:
                        global_mask = get_pruning_mask(_shape[0],  global_indices)
                        # set_param(self, _n, nn.Parameter(_p.data[global_mask]))
                        _v['exp_avg'] = _v['exp_avg'][global_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][global_mask]
                    else:
                        global_mask = get_pruning_mask(_shape[0],  global_indices)
                        local_mask = get_pruning_mask(_shape[1],  local_indices)
                        # new_p = _p.data[global_mask, :][:, local_mask]
                        # set_param(self, _n, nn.Parameter(new_p.data))
                        _v['exp_avg'] = _v['exp_avg'][global_mask,:][:,local_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][global_mask,:][:,local_mask]
                else:
                    if 'bias' in _n:
                        local_mask = get_pruning_mask(_shape[0],  local_indices)
                        # set_param(self, _n, nn.Parameter(_p.data[local_mask]))
                        _v['exp_avg'] = _v['exp_avg'][local_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][local_mask]
                    else:
                        global_mask = get_pruning_mask(_shape[1],  global_indices)
                        local_mask = get_pruning_mask(_shape[0],  local_indices)
                        # new_p = _p.data[local_mask, :][:, global_mask]
                        # set_param(self, _n, nn.Parameter(new_p.data))
                        _v['exp_avg'] = _v['exp_avg'][local_mask,:][:,global_mask]
                        _v['exp_avg_sq'] = _v['exp_avg_sq'][local_mask,:][:,global_mask]
            _dict[param_list[_i]] = _v
            _i+=1
        self.state = _dict
    '''
    def pruning(self, gl_dict, _model, eps=1e-8):
        en_heads = _model.cfg.encoder.attention_heads
        de_heads = _model.cfg.decoder.attention_heads

        named_params = list(_model.named_parameters())
        param_list = list(_model.parameters())

        # model_params = list(param_list)
        self.param_groups[0]['params'] = param_list

        # named_params = list(_model.named_parameters())
        _dict = {}
        
        for _i, (_k, _v) in enumerate(self.state.items()):
            _n = named_params[_i][0]
            if 'embed_tokens' in _n or 'layer_norm' in _n or 'alpha' in _n:
                pass
            elif '_c' in _n:
                pass
            else:
                ende, ly, type, wb = _parsing(_n)
                num_heads = en_heads if ende=='encoder' else de_heads
                
                if 'proj' in type:
                    attn_type = type.split('.')[0]
                    if 'q_proj' in type or 'k_proj' in type:
                        # qk proj
                        _key = f'{ende}.{ly}.{attn_type}.qk'
                    else:
                        # vo proj
                        _key = f'{ende}.{ly}.{attn_type}.vo'
                    _gl, _count = gl_dict[_key]
                    _mask = ((_gl/_count)>eps).repeat(num_heads)

                    if 'out_proj' in type:
                        if 'weight' in wb:
                            _v['exp_avg'] = _v['exp_avg'][:, _mask]
                            _v['exp_avg_sq'] = _v['exp_avg_sq'][:, _mask]
                        else:
                            pass
                    else:
                        # q, k, v proj
                        if 'weight' in wb:
                            _v['exp_avg'] = _v['exp_avg'][_mask, :]
                            _v['exp_avg_sq'] = _v['exp_avg_sq'][_mask, :]
                        else:
                            _v['exp_avg'] = _v['exp_avg'][_mask]
                            _v['exp_avg_sq'] = _v['exp_avg_sq'][_mask]
                elif 'fc' in type:
                    _key = f'{ende}.{ly}.fc'
                    _gl, _count = gl_dict[_key]
                    _mask = (_gl/_count)>eps
                    if 'fc1' in type:
                        if 'weight' in wb:
                            _v['exp_avg'] = _v['exp_avg'][_mask, :]
                            _v['exp_avg_sq'] = _v['exp_avg_sq'][_mask, :]
                        else: 
                            _v['exp_avg'] = _v['exp_avg'][_mask]
                            _v['exp_avg_sq'] = _v['exp_avg_sq'][_mask]
                    else:
                        # fc2
                        if 'weight' in wb:
                            _v['exp_avg'] = _v['exp_avg'][:, _mask]
                            _v['exp_avg_sq'] = _v['exp_avg_sq'][:, _mask]
                        else:
                            pass
            _dict[param_list[_i]] = _v
        self.state = _dict
    '''                     
                    

                            


                        

