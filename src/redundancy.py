#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse, time
import logging, pickle
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import checkpoint_utils
from fairseq import options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
# from fairseq.criterions.spt import get_group_sum, group_report


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)

    ############# Perform shaping Model for loading Pruned Model #################
    # For spt
 
    _src_words = model.encoder.embed_tokens.weight.shape[0]
    _tar_words = model.decoder.embed_tokens.weight.shape[0]
    pm = PruningManager(cfg, _src_words, _tar_words)

    setattr(model, "pruning_manager", pm)

    # pass checkpoint path and shaving model
    pretrained_model = f'{cfg.checkpoint.save_dir}/{cfg.checkpoint.restore_file}'
    if os.path.isfile(pretrained_model):
    # if cfg.checkpoint.restore_file == 'checkpoint_best.pt':
        print("+++++++ Loading pre-trained model for finetuning +++++++")
        model = checkpoint_utils.load_spt(pretrained_model, model)
        print("+++++ Loading pre-trained model for finetuning done +++++")
    ##############################################################################
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator

    ##################### For SPT ################################
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    trainer.model.pruning_manager.encoder_indices = torch.arange(512)
    trainer.model.pruning_manager.decoder_indices = torch.arange(512)
    ##############################################################
    def _shape(_tensor):
        _r = "[" + ", ".join([f'{i}' for i in _tensor.shape]) +"]"
        return _r
        
    ##############################################################
    # print(trainer.model.encoder.embedding_c)
    print("########### HELLO ##################")
    """
    with open('sample', 'rb') as ff:
        samples = pickle.load(ff)
    """
    trainer.model.eval()
    # print(trainer.model.encoder.embedding_c)  
    _sd = model.state_dict()

    S_dict = {}
    _frac = 0.90
    for ende in ['encoder', 'decoder']:
        print(f"------------------------ {ende} -------------------------------")
        for _ly in range(0, 6):
            _n1 = f'{ende}.layers.{_ly}.self_attn.q_proj.weight'
            _n2 = f'{ende}.layers.{_ly}.self_attn.k_proj.weight'
            Q = _sd[_n1]
            K = _sd[_n2]
            head_dim_qk = Q.shape[0] // 4
            head_dim_embed = Q.shape[1] // 4
            print(f"* Layer: {_ly} | Q shape: {_shape(Q)} | head_dim_qk: {head_dim_qk} | head_dim_embed: {head_dim_embed}")
            qTks = []
            for h in range(0,4):
                # p1 = Q[head_dim_qk*h: head_dim_qk*(h+1), head_dim_embed*h: head_dim_embed*(h+1)]
                # p2 = K[head_dim_qk*h: head_dim_qk*(h+1), head_dim_embed*h: head_dim_embed*(h+1)]
                p1 = Q[head_dim_qk*h: head_dim_qk*(h+1), :]
                p2 = K[head_dim_qk*h: head_dim_qk*(h+1), :]
                qTk = torch.mm(p1.T, p2)
                qTks.append(qTk)
                print(f"** p1.shape: {_shape(p1)} | p2.shape: {_shape(p2)} | p1.T p2 .shape: {_shape(qTk)}")
                U, S, Vt = torch.linalg.svd(qTk)
                # U, S = torch.eig(qTk, eigenvectors=True)
                # S = torch.diag(S,0)

                _s = torch.sum(S)
                S /= _s
                # print(S)
                S_dict[f'{ende}.{_ly}.qk'] = S
                _acc = 0.
                for i, ss in enumerate(S):
                    _acc += ss
                    if _acc > _frac:
                        print(f"{_frac * 100:.0f}% | {ende} | {_ly} | {h} | {i}")
                        break

    # for _k in S_dict:
    #     print(_k, "\n", S_dict[_k])
        

    time.sleep(1000)
    ##############################################################

class PruningManager():
    def __init__(self, cfg, src_words, tar_words): 
        self.src_words = src_words
        self.tar_words = tar_words
        
        self.en_layers = cfg.model.encoder_layers
        self.de_layers = cfg.model.decoder_layers

        self.en_heads = cfg.model.encoder_attention_heads
        self.de_heads = cfg.model.decoder_attention_heads

        self.GLE = cfg.model.encoder_embed_dim
        self.GLD = cfg.model.decoder_embed_dim
        self.FC = cfg.model.encoder_ffn_embed_dim * self.en_layers \
                + cfg.model.decoder_ffn_embed_dim * self.de_layers
        self.QK = cfg.model.encoder_embed_dim // self.en_heads * self.en_layers \
                + cfg.model.decoder_embed_dim // self.de_heads * self.de_layers * 2 
        self.VO = cfg.model.encoder_embed_dim // self.en_heads * self.en_layers \
                + cfg.model.decoder_embed_dim // self.de_heads * self.de_layers * 2 

        self.count = 0
        self._count = 0

        self.pruning_dict = None

        # Compute pruing ratio    
        self.step_per_epoch = 1102 #TODO: step per epoch is hard-coded
        self.pruning_iter = cfg.model.pruning_iter
        self.pruning_period = cfg.model.pruning_period
        self.warming_up = cfg.model.warming_up
        self.compression_rate = cfg.model.compression_rate

        # self.P = 1 - np.sqrt(cfg.common.compression_rate)
        self.P = 1-self.get_pruning_rate(self.compression_rate)
        self.p = 1 - (1-self.P) ** (1/self.pruning_iter) 

        # For positional embedding
        self.encoder_orig_dim = cfg.model.encoder_embed_dim
        self.decoder_orig_dim = cfg.model.decoder_embed_dim

        self.encoder_indices = torch.arange(self.encoder_orig_dim)
        self.decoder_indices = torch.arange(self.decoder_orig_dim)
        
        _decreasing = cfg.model.decreasing
        if _decreasing == 'epochwise':
            # Epoch-wise shrink
            self.c_shrink_rate =  1 / (cfg.model.pruning_period)
        else:
            # Step-wise shrink (difference)
            self.c_shrink_rate =  ((1-1e-5) / (cfg.model.pruning_period * self.step_per_epoch-1))

        # Step-wise shrink (ratio)
        # self.c_shrink_rate = (1e-5) ** (1/ (cfg.common.pruning_period * self.step_per_epoch))


    def get_pruning_rate(self, compression_rate):
        assert self.GLE == self.GLD # For simplify computation of n1
        assert self.en_heads == self.de_heads

        n1 = float(self.GLE * (self.FC * 2 + self.QK * self.en_heads * 2 + self.VO * self.en_heads * 2))
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


    def get(self, ):
        assert self.count < self.pruning_iter
        self.count += 1
        gle = int(np.ceil(self.GLE * self.p))
        gld = int(np.ceil(self.GLD * self.p))
        fc = int(np.ceil(self.FC * self.p))
        qk = int(np.ceil(self.QK * self.p))
        vo = int(np.ceil(self.VO * self.p))

        self.GLE -= gle
        self.GLD -= gld
        self.FC -= fc
        self.QK -= qk
        self.VO -= vo
        
        return gle, gld, fc, qk, vo

    def update_embedding_mask(self, emb_mask, ende):
        # emb_mask: indices for pruning (len: pruned_len)
        # self.encoder_indices: len: orig_dim
        # 0, 1, 2, ...... , 511
        if ende == 'encoder':
            emb_mask2 = torch.zeros(self.GLE + emb_mask.shape[0]).bool()
            emb_mask2[emb_mask] = True
            self.encoder_indices.data = self.encoder_indices[~emb_mask2]
        else:
            emb_mask2 = torch.zeros(self.GLD + emb_mask.shape[0]).bool()
            emb_mask2[emb_mask] = True
            self.decoder_indices.data = self.decoder_indices[~emb_mask2]

    def get_embedding_mask(self, ende, _dev='cpu'):
        if ende == 'encoder':
            _mask = torch.zeros(self.encoder_orig_dim).bool()
            _mask[self.encoder_indices] = True
        else:
            _mask = torch.zeros(self.decoder_orig_dim).bool()
            _mask[self.decoder_indices] = True
        _mask = _mask.to(_dev)
        return _mask
  
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
                        ((e - self.warming_up) % self.pruning_period == 1) \
                        and (e - self.warming_up) -  (self.pruning_period * self.pruning_iter) <= 1
        return _p, do_pruning

def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
