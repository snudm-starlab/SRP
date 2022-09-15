# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math, time
from dataclasses import dataclass, field
import numpy as np

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SPTConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    
    sentence_avg: bool = II("optimization.sentence_avg")

    # For SPT
    local_qk_gl: float = field(
        default=0.3,
        metadata={"help": "coefficient for local group lasso regularization for attns_qk"},
    )
    local_vo_gl: float = field(
        default=0.3,
        metadata={"help": "coefficient for local group lasso regularization for attns_vo"},
    )
    local_fc_gl: float = field(
        default=0.3,
        metadata={"help": "coefficient for local group lasso regularization for fc layers"},
    )
    global_gl: float = field(
        default=10,
        metadata={"help": "coefficient for global group lasso regularization"},
    )

    reg: float = field(
        default=10,
        metadata={"help": "coefficient for l2 regularization"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


# For SPT
'''
def get_group_sum(model):
    gl_dict = {} # _key: [_gl, _count]
    en_heads = model.cfg.encoder.attention_heads
    de_heads = model.cfg.decoder.attention_heads
    for _n, _p in model.named_parameters():
        if 'embed_tokens' in _n:
            continue
        elif '_c' in _n:
            continue
        elif 'layer_norm' in _n or 'alpha' in _n:
            _key = 'global'
            _gl = _p * _p
            _count = _p.shape[0]
        else:
            ende, ly, type, wb = _parsing(_n)
            num_heads = en_heads if ende=='encoder' else de_heads
            if 'proj' in type:
                attn_type = type.split('.')[0]
                if 'k_proj' in type or 'q_proj' in type:
                    _key = f'{ende}.{ly}.{attn_type}.qk'
                else:
                    _key = f'{ende}.{ly}.{attn_type}.vo'
                if 'out_proj' in type:
                    if 'weight' in wb:
                        _head_dim = _p.shape[1] // num_heads
                        _count = _p.shape[0] * num_heads
                        _tmp = torch.sum(_p*_p, dim=0)
                        _gl = _tmp[0: _head_dim]
                        for i in range(1, num_heads):
                            _gl += _tmp[i*_head_dim: (i+1)*_head_dim]                 

                        # For global_gl
                        """
                        _global_count = _p.shape[1] * 2
                        _tmp2 = torch.sum(_p*_p, dim=1)
                        _global_gl = _tmp2[::2] + _tmp2[1::2]
                        """
                    else:
                        _count = -1 # Flag for skipping _gl computation
                        """
                        _global_count = 2
                        _tmp2 = _p*_p
                        _global_gl = _tmp2[::2] + _tmp2[1::2]
                        """

                else:
                    # 'q,k,v_proj'
                    if 'weight' in wb:
                        _head_dim = _p.shape[0] // num_heads
                        _count = _p.shape[1] * num_heads
                        _tmp = torch.sum(_p*_p, dim=1)
                        
                        """
                        # For global_gl
                        _global_count = _p.shape[0] * 2
                        _tmp2 = torch.sum(_p*_p, dim=0)
                        _global_gl = _tmp2[::2] + _tmp2[1::2]
                        """

                    else:
                        _head_dim = _p.shape[0] // num_heads
                        _count = num_heads
                        _tmp = _p * _p 
                        """
                        _global_count = -1
                        """
                    _gl = _tmp[0: _head_dim]
                    for i in range(1, num_heads):
                        _gl += _tmp[i*_head_dim: (i+1)*_head_dim]

            elif 'fc' in type:
                _key = f'{ende}.{ly}.fc'
                if 'fc1' in type:
                    if 'weight' in wb:
                        _count = _p.shape[1]
                        _gl = torch.sum(_p * _p, dim=1)

                        """
                        # For global_gl
                        _global_count = _p.shape[0] * 2
                        _tmp2 = torch.sum(_p * _p, dim=0)
                        _global_gl = _tmp2[::2] + _tmp2[1::2]
                        """
                    else:
                        _count = 1
                        _gl = _p*_p
                        """
                        _global_count = -1
                        """
                elif 'fc2' in type:
                    # fc2
                    if 'weight' in wb:
                        _count = _p.shape[0]
                        _gl = torch.sum(_p*_p, dim=0)
                        """
                        # For global_gl
                        _global_count = _p.shape[1] * 2
                        _tmp2 = torch.sum(_p * _p, dim=1)
                        _global_gl = _tmp2[::2] + _tmp2[1::2]
                        """
                    else:
                        _count = -1 # Flag for skipping local_gl computation
                        """
                        # For global_gl
                        _global_count = _p.shape[0]
                        _tmp2 = _p * _p
                        _global_gl = _tmp2[::2] + _tmp2[1::2]
                        """
                else:
                    _count = -1
            else:
                print("Unknwon parameter found!")


        # gl_dict -> k:v = local_key: [local_gl, local_count]
        if _count == -1:
            pass
        elif _key in gl_dict:
            gl_dict[_key][0] += _gl
            gl_dict[_key][1] += _count
        else:
            gl_dict[_key] = [_gl, _count]
        """
        # Update global_gl and global_count
        if global_count == -1:
            pass
        elif global_gl is not None and global_count is not None:
            global_gl += _global_gl
            global_count += _global_count
        else:
            global_gl = _global_gl
            global_count = _global_count
        """
    # return local_gl_dict, global_gl, global_count
    return gl_dict


def group_report(model, gl_dict, eps=1e-8):
    print("="*15, 'PRUNING STATUS' ,"="*15)
    _params = get_param_num(model)
    print("* Number of Paramters: ", _params)
    _res = ''
    for _k in gl_dict.keys():
        _suv = torch.sum(gl_dict[_k][0]/gl_dict[_k][1] > eps)
        _res += f'{_suv},'
        if 'global' in _k:
            print(f"{_k}: {_suv}/512 | count: {gl_dict[_k][1]}") 
            print("Global gl[0:20]: ")
            print(gl_dict[_k][0][0:20])
        elif 'fc' in _k:
            print(f"{_k}: {_suv}/1024 | count: {gl_dict[_k][1]}")
        else:
            print(f"{_k}: {_suv}/128 | count: {gl_dict[_k][1]}")
    print("="*50)
    _res +=f'{_params}'
    return _res


def get_param_num(model):
    _num = np.sum([_p.numel() for _p in model.parameters()])
    return _num 


def group_lasso_loss(model):
    # Compute local_gl and global_gl
    gl_dict = get_group_sum(model)

    global_gl_loss = None 
    # local_attn_gl_loss, local_fc_gl_loss = None, None
    local_qk_gl_loss, local_vo_gl_loss, local_fc_gl_loss = None, None, None
    for _k in gl_dict.keys():
        _gl, _count = gl_dict[_k]
        _gl_loss = torch.sum(torch.sqrt(_gl) / \
                         torch.sqrt(torch.tensor(_count)))
        if 'global' in _k:
            global_gl_loss = _gl_loss
        elif 'qk' in _k:
            if local_qk_gl_loss is None:
                local_qk_gl_loss = _gl_loss
            else:
                local_qk_gl_loss += _gl_loss
        elif 'vo' in _k:
            if local_vo_gl_loss is None:
                local_vo_gl_loss = _gl_loss
            else:
                local_vo_gl_loss += _gl_loss
        else:
            if local_fc_gl_loss is None:
                local_fc_gl_loss = _gl_loss
            else:
                local_fc_gl_loss += _gl_loss

    # local_gl
    # global_gl
    # global_gl_loss = torch.sum(torch.sqrt(global_gl) / torch.sqrt(torch.tensor(global_count)))
    # print("Local: ",local_gl_loss, "Global: ", global_gl_loss)
    # time.sleep(10000)
    return local_qk_gl_loss, local_vo_gl_loss, local_fc_gl_loss, global_gl_loss
'''

def _parsing(_name):
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


def reg_loss(param, reg_type='l1'):
    if reg_type == 'l1':
        return torch.sum(torch.abs(param))
    elif reg_type == 'l2':
        return torch.sum(param * param)
    else:
        return None

def update_loss(loss, new_loss):
    if loss is None:
        loss = new_loss
    else:
        loss += new_loss
    return loss


def get_reg_loss(model, reg_type='l1'):
    pm = model.pruning_manager
    pd = pm.pruning_dict

    en_heads = model.cfg.encoder.attention_heads
    de_heads = model.cfg.decoder.attention_heads
    _loss = None
    for _n, _p in model.named_parameters():
        
        if _n[-2:] == "_c" or 'embed_tokens' in _n:
            # skip connection parameters
            continue  
        elif 'alpha' in _n:
            ende = _n.split('.')[0]
            _key = f"{ende}.embedding_c"
            mask = pd[_key]
            new_loss = reg_loss(_p[mask], reg_type=reg_type) # torch.sum( _p[mask] * _p[mask])
            _loss = update_loss(_loss, new_loss)

        
        elif 'layer_norm' in _n:
            ende, ly, type, wb = _parsing(_n)
            if 'self' in type:
                _type = 'self_attn'
            elif 'encoder' in type:
                _type = 'encoder_attn'
            else:
                _type = 'fc'
            _key = f"{ende}.layers.{ly}.{_type}_ln_c"
            mask = pd[_key] if _key in pd else []
            new_loss = reg_loss(_p[mask], reg_type=reg_type) # torch.sum(_p[mask] * _p[mask])
            _loss = update_loss(_loss, new_loss)
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

            global_mask = pd[global_key] if global_key in pd else []
            local_mask = pd[local_key] if local_key in pd else []


            if 'fc2' in _n:
                if 'bias' in _n:
                    new_loss = reg_loss(_p[global_mask], reg_type=reg_type)  
                    # torch.sum(_p[global_mask] * _p[global_mask])
                else:
                    loss_1 = reg_loss(_p[global_mask, :], reg_type=reg_type)
                    # torch.sum(_p[global_mask,:] ** 2)
                    loss_2 = reg_loss(_p[:, local_mask], reg_type=reg_type)
                    # torch.sum(_p[:,local_mask] ** 2)
                    loss_3 = reg_loss(_p[global_mask, :][:, local_mask], reg_type=reg_type)
                    # torch.sum(_p[global_mask,:][:,local_mask] ** 2)
                    new_loss = loss_1 + loss_2 - loss_3 
            else:
                if 'bias' in _n:
                    new_loss = reg_loss(_p[local_mask], reg_type=reg_type)
                    # torch.sum(_p[local_mask] * _p[local_mask])
                else:
                    # print(f"* {_n}: {_p.shape} | {local_mask} {local_mask.shape}")
                    # print(_n)
                    # print(_p.shape)
                    # print(local_mask)
                    loss_1 = reg_loss(_p[:, global_mask], reg_type=reg_type)
                    # torch.sum(_p[:,global_mask] ** 2)
                    loss_2 = reg_loss(_p[local_mask, :], reg_type=reg_type)
                    # torch.sum(_p[local_mask,:] ** 2)
                    loss_3 = reg_loss(_p[local_mask,:][:,global_mask], reg_type=reg_type)
                    # torch.sum(_p[local_mask,:][:,global_mask] ** 2)
                    new_loss = loss_1 + loss_2 - loss_3
            _loss = update_loss(_loss, new_loss)
            # print(" ******* ", _n, " Ends")

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
            # o: (gl_dim, vo_dim) | bias: gl_dim | global: previous sub-layer ln_c
            
            if global_key in pd:
                global_mask = pd[global_key]
            else:
                global_mask = []
            if local_key in pd:
                local_mask = pd[local_key]
            else:
                local_mask = []

            # Compute loss 
            if 'out_proj' in _n:
                if 'bias' in _n:
                    new_loss = reg_loss(_p[global_mask], reg_type=reg_type)
                    #  torch.sum(_p[global_mask] * _p[global_mask])
                else:
                    loss_1 = reg_loss(_p[global_mask, :], reg_type=reg_type)
                    # torch.sum(_p[global_mask,:] ** 2)
                    loss_2 = reg_loss(_p[:, local_mask], reg_type=reg_type)
                    # torch.sum(_p[:,local_mask] ** 2)
                    loss_3 = reg_loss(_p[global_mask, :][:, local_mask], reg_type=reg_type)
                    # torch.sum(_p[global_mask,:][:,local_mask] ** 2)
                    new_loss = loss_1 + loss_2 - loss_3 
                    # new_loss = loss_1 + loss_2
            else:
                if 'bias' in _n:
                    new_loss = reg_loss(_p[local_mask], reg_type=reg_type)
                    # torch.sum(_p[local_mask] * _p[local_mask])
                else:
                    loss_1 = reg_loss(_p[:, global_mask], reg_type=reg_type)
                    # torch.sum(_p[:,global_mask] ** 2)
                    loss_2 = reg_loss(_p[local_mask, :], reg_type=reg_type)
                    # torch.sum(_p[local_mask,:] ** 2)
                    loss_3 = reg_loss(_p[local_mask,:][:,global_mask], reg_type=reg_type)
                    # torch.sum(_p[local_mask,:][:,global_mask] ** 2)
                    new_loss = loss_1 + loss_2 - loss_3
                    # new_loss = loss_1 + loss_2
            _loss = update_loss(_loss, new_loss)
    return _loss



# For SPT end

@register_criterion(
    "spt", dataclass=SPTConfig
)
class SPTCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True, scoring=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], scoring=scoring)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, scoring=scoring)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, scoring=False):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        phase = getattr(model, 'phase', 'x')
        if phase == 'pruning' and not scoring:
            loss += model.cfg.reg * get_reg_loss(model, reg_type='l1')
            
            '''
            local_qk_gl_loss, local_vo_gl_loss, local_fc_gl_loss, global_gl_loss = group_lasso_loss(model)

            loss += model.cfg.local_qk_gl * local_qk_gl_loss + \
                    model.cfg.local_vo_gl * local_vo_gl_loss + \
                    model.cfg.local_fc_gl * local_fc_gl_loss + \
                    model.cfg.global_gl * global_gl_loss
            '''
            
        """
        return local_attn_gl_loss, local_fc_gl_loss, global_gl_loss
        print(f"loss: {loss} | "
              f"nll_loss: {nll_loss} | local_gl_loss: {local_gl_loss*model.cfg.local_gl}"
              f" | global_gl_loss: {global_gl_loss * model.cfg.global_gl}")
        """
        # time.sleep(1000)

        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
