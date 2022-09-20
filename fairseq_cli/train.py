#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse, time
import logging
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
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
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

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
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
    """ # Original Code #
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    """
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    ##############################################################


    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    
    ######################## For STP #######################
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

            self.pruning_dict = None

            # Compute pruing ratio
            self.step_per_epoch = 1102 #TODO: step per epoch is hard-coded
            self.n = cfg.common.pruning_iter
            self.compression_rate = cfg.common.compression_rate
        
            # self.P = 1 - np.sqrt(cfg.common.compression_rate)
            self.P = 1-self.get_pruning_rate(self.compression_rate)
            self.p = 1 - (1-self.P) ** (1/self.n) 

            # For positional embedding
            self.encoder_orig_dim = cfg.model.encoder_embed_dim
            self.decoder_orig_dim = cfg.model.decoder_embed_dim

            self.encoder_indices = torch.arange(self.encoder_orig_dim)
            self.decoder_indices = torch.arange(self.decoder_orig_dim)
            
            # For same percentage
            # self.c_shrink_rate = (1e-5) ** (1/ (cfg.common.pruning_period * self.step_per_epoch))
            # For same difference
            # self.c_shrink_rate =  ((1-1e-5) / (cfg.common.pruning_period * self.step_per_epoch-1))
            # shirnk c every epoch
            self.c_shrink_rate =  (1 / (cfg.common.pruning_period-1))

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
            assert self.count < self.n
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
                self.encoder_indices = self.encoder_indices[~emb_mask2]
            else:
                emb_mask2 = torch.zeros(self.GLD + emb_mask.shape[0]).bool()
                emb_mask2[emb_mask] = True
                self.decoder_indices = self.decoder_indices[~emb_mask2]

        def get_embedding_mask(self, ende, _dev='cpu'):
            if ende == 'encoder':
                _mask = torch.zeros(self.encoder_orig_dim).bool()
                _mask[self.encoder_indices] = True
            else:
                _mask = torch.zeros(self.decoder_orig_dim).bool()
                _mask[self.decoder_indices] = True
            _mask = _mask.to(_dev)
            return _mask
 
    _src_words = trainer.model.encoder.embed_tokens.weight.shape[0]
    _tar_words = trainer.model.decoder.embed_tokens.weight.shape[0]
    pm = PruningManager(cfg, _src_words, _tar_words)
    setattr(trainer.model, "pruning_manager", pm)
    setattr(trainer.model, 'phase', 'warming-up')

    pruning_count = 0
    # phase: 'pruning' or 'fine-tuning'
    #########################################################

    while epoch_itr.next_epoch_idx <= max_epoch:
        phase = getattr(trainer.model, 'phase')
        if (phase == 'warming-up') and (epoch_itr.epoch >= cfg.common.warming_up):
            setattr(trainer.model, 'phase', 'pruning')
            setattr(pm, '_count', 0)
            phase = getattr(trainer.model, 'phase')
            print(f"* End of {cfg.common.warming_up} warming-up epochs. Turn into 'pruning' phase")
            # print(f"\n**** {epoch_itr.epoch+1}/{cfg.common.warming_up} | change phase 'warming-up' to 'pruning'\n")
        

        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr, pruning_manager=pm)
        # Check pruning target
        _params = np.sum([_p.numel() for _n, _p in trainer.model.named_parameters()
                          if _n[-2:] != '_c'])
        num_groups = trainer.model.get_num_groups()
        num_groups = [str(_num) for _num in num_groups]
        
        ##################### SPT  Pruning ##########################
        # gl_dict = get_group_sum(trainer.model) 
        if phase == 'pruning':
            # Get Group sum
            # _eps = cfg.common.prune_eps
            # local_gl_dic --> k:v = local_key: [local_gl, local_count]
            # pm._count += 1 # regularization without pruning
            setattr(pm, '_count', 
                    (epoch_itr.epoch - cfg.common.warming_up)%cfg.common.pruning_period)
            if pm._count == 0:
                print(f"\n**** {epoch_itr.epoch} | Perform pruning #{pruning_count+1}'\n")
                trainer.model.pruning()
                trainer.optimizer._optimizer.pruning(trainer.model)
                pm.update_embedding_mask(
                    pm.pruning_dict['encoder.embedding_c'], 'encoder')
                pm.update_embedding_mask(
                    pm.pruning_dict['decoder.embedding_c'], 'decoder')
                pruning_count += 1
                # print(f"# params: {_params}")
                if pruning_count == cfg.common.pruning_iter:
                    setattr(trainer.model, 'phase', 'fine-tuning')
                # pm._count = 0


        # print pruning status
        
        _res = f'{phase[0]},{epoch_itr.epoch},'
        _res+= ','.join(num_groups) + ','
        # _group_res = group_report(trainer.model, gl_dict)
        # _res += _group_res
        _res += f'{_params},{valid_losses[0]}'
        print("+"*15, '  Test ', '+'*15)
        print(_res)
        _path_list = cfg.checkpoint.save_dir.split('/')
        _res_file = f'./checkpoints/res_files/{_path_list[-1]}.csv'
        print("Result file:", _res_file)
        print("+"*15, '  Test ', '+'*15)
        with open(_res_file, 'a') as f:
            f.write(_res + '\n')
        
        # Save pruning status (param/ bleu/ groups change)
        ##############################################################

        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, pruning_manager=None,
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    # epoch_itr.epoch increased
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    
    # if cfg.common.tpu:
    #     itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    if trainer.model.phase == 'pruning':
        if getattr(pruning_manager, "_count", "No_Count") == 0:
            scoring=True
        else:
            scoring=False
            trainer.model.decrease_c(ratio=pruning_manager.c_shrink_rate)
    else:
        scoring=False

    for i, samples in enumerate(progress):
        """
        # For SPT, decrease connecivity parameters
        if trainer.model.phase == "pruning" and not scoring:
            assert pruning_manager.pruning_dict is not None
        """
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
        

            log_output = trainer.train_step(samples, scoring=scoring)

        if scoring:
            # Scoring groups at the beginning of every epoch
            gle, gld, fc, qk, vo = pruning_manager.get()
            print("#"*85)
            print(f"# groups to remove: GLE ({gle}) | GLD ({gld}) | FC ({fc})  | QK ({qk}) | VO ({vo})#")
            # scoring_groups(trainer.model)
            pruning_dict = {}
            pruning_dict.update(
                get_fc_dict(trainer.model, fc)
            )
            pruning_dict.update(
                get_global_dict(trainer.model, gle, "encoder")
            )
            pruning_dict.update(
                get_global_dict(trainer.model, gld, "decoder")
            )
            pruning_dict.update(
                get_qkvo_dict(trainer.model, qk, "qk")
            )
            pruning_dict.update(
                get_qkvo_dict(trainer.model, qk, "vo")
            )

            # for k in pruning_dict:
            #     print(k, type(pruning_dict[k]))
            pruning_manager.pruning_dict = pruning_dict           
            print("#"*75)

            trainer.model.zero_grad()
            trainer.zero_grad()
            scoring = False
            # trainer.model.decrease_c(ratio=pruning_manager.c_shrink_rate)
            continue


        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
        """    
        ################# SPT  Pruning (Step-wise) ###################
        # Perform pruning
        # Get Group sum
        gl_dict = get_group_sum(trainer.model)
        
        _eps = cfg.common.prune_eps
        # local_gl_dic --> k:v = local_key: [local_gl, local_count]
        trainer.model.pruning(gl_dict,  eps=_eps)
        trainer.optimizer._optimizer.pruning(gl_dict, trainer.model, eps=_eps) 
        ##############################################################
        """
        
        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config

############### Scoring groups for SPT ######################
'''
def scoring_groups(model):
    print(" =============== Scoring ===============")
    _dict = dict(model.named_parameters())
    for _n, _p in model.named_parameters():
        if 'qk_c' in _n or 'vo_c' in _n:
            print("*", _n, ": ", torch.mean(_p.grad))
'''

def get_global_dict(model, gl, ende):
    gl_dict = {} # k:v = param_name: pruning indices
    module_list = ["self_attn", "fc"]
    if ende == "decoder":
        module_list.append("encoder_attn")
    score_sum = None
    _names = []
    for module in module_list:
        for ly in range(6):
            _n = f"{ende}.layers.{ly}.{module}_ln_c"
            score = get_attr(model, _n).grad
            if score is not None:
                if score_sum is None:
                    score_sum = score
                else:
                    score += score_sum
                _names.append(_n)
                # gl_dict[_n] = torch.argsort(score, descending=True)[:gl]
            # print(_n, gl, torch.argsort(score)[:gl])

    _n = f"{ende}.embedding_c"
    score = get_attr(model, _n)
    if score is not None:
        if score_sum is None:
            score_sum = score
        else:
            score_sum += score
        _names.append(_n)

    _mask = torch.argsort(score_sum, descending=True)[:gl]
    for _n in _names:
        gl_dict[_n] = _mask.clone()
    del _mask

    # gl_dict[_n] = torch.argsort(score, descending=True)[:gl]
    return gl_dict

"""
def get_global_dict(model, gl, ende):
    gl_dict = {} # k:v = param_name: pruning indices
    module_list = ["self_attn", "fc"]
    if ende == "decoder":
        module_list.append("encoder_attn")
    for module in module_list:
        for ly in range(6):
            _n = f"{ende}.layers.{ly}.{module}_ln_c"
            score = get_attr(model, _n).grad
            if score is not None:
                gl_dict[_n] = torch.argsort(score, descending=True)[:gl]
            # print(_n, gl, torch.argsort(score)[:gl])
    _n = f"{ende}.embedding_c"
    score = get_attr(model, _n)
    gl_dict[_n] = torch.argsort(score, descending=True)[:gl]
    return gl_dict
"""


def get_qkvo_dict(model, pn, qkvo):
    pruning_dict = {} # k:v = param_name: pruning indices
    score_dict = {} # k:v = param_name: score, args_list
    scores = []
    for ende in ["encoder", "decoder"]:
        module_list = ["self_attn"]
        if ende == "decoder":
            module_list.append("encoder_attn")
        for ly in range(6):
            for module in module_list:
                _n = f"{ende}.layers.{ly}.{module}_{qkvo}_c"
                _grad = get_attr(model, _n).grad
                if _grad is None:
                    continue
                _head_dim = _grad.shape[-1]//4
                score = None
                # print(f"### _grad shape: {_grad.shape[0]} | _head_dim: {_head_dim}")
                args_list = []
                for i in range(4):
                    temp_grad, _args = torch.sort(_grad[_head_dim*i:_head_dim*(i+1)], descending=True)
                    # print(f"**** {i}: ", temp_grad.shape, _head_dim)
                    _args += _head_dim * i
                    args_list.append(_args)
                    if score == None:
                        score = temp_grad
                    else:
                        score += temp_grad
                score_dict[_n] = [score, args_list]
                scores.append(score)
    # print("Length of scores: ", len(scores))
    scores = torch.sort(torch.cat(scores), descending=True)[0]
    # print("Shape of scores: ", scores.shape)
    thres = scores[pn]                  
    # print(thres)
    # print("* Pruning: ", pn)
    count = 0
    for ende in ["encoder", "decoder"]:
        module_list = ["self_attn"]
        if ende == "decoder":
            module_list.append("encoder_attn")
        for ly in range(6):
            for module in module_list:
                _n = f"{ende}.layers.{ly}.{module}_{qkvo}_c"
                if _n in score_dict:
                    score, _args = score_dict[_n]
                    _head_dim = score.shape[-1]
                    cond = (score > thres)
                    if torch.sum(cond) > 0:
                        _inds = torch.where(cond)[0]
                        count += _inds.shape[0]
                        _inds_all = torch.cat([ _ind[cond] for _ind in _args])
                        pruning_dict[_n] = _inds_all
    # print("Count: ", count)
    return pruning_dict     


def get_fc_dict(model, fc):
    fc_dict = {} # k:v = param_name: pruning indices
    scores = []
    for ende in ['encoder', 'decoder']:
        for ly in range(0,6):
            _n = f'{ende}.layers.{ly}.fc_c'
            score = get_attr(model, _n).grad
            if score is None:
                continue
            scores.append(score)
            # print(_n, score.shape)
    scores = torch.sort(torch.cat(scores), descending=True)[0]
    # print(scores.shape)
    # print(scores[0:50])
    # print(scores[-50:])
    thres = scores[fc]
    # print(thres)
    # print("###############################")
    # print("* fc: ", fc)
    count = 0
    for ende in ['encoder', 'decoder']:
        for ly in range(0,6):
            _n = f'{ende}.layers.{ly}.fc_c'
            score = get_attr(model, _n).grad
            if score is None:
                continue
            cond = (score > thres)
            if torch.sum(cond) > 0:
                # print(_n, torch.where(cond)[0])
                count += torch.where(cond)[0].shape[0]
                fc_dict[_n] = torch.where(cond)[0]
    # print("* count: ", count)
    return fc_dict


def get_attr(_model, _name):
    _attrs = _name.split(".")
    _parent = _model
    for _attr in _attrs[:-1]:
        _parent = getattr(_parent, _attr)
    return getattr(_parent, _attrs[-1])
            
##############################################################

def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


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
