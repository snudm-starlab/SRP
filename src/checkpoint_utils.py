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

import ast
import collections
import contextlib
import inspect
import logging
import os
import re
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from fairseq import meters, tasks
from fairseq.data import data_utils
from fairseq.dataclass.configs import CheckpointConfig
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP
from fairseq.file_io import PathManager
from fairseq.models import FairseqDecoder, FairseqEncoder
from omegaconf import DictConfig, OmegaConf, open_dict, _utils

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise ImportError(
        "You need to install huggingface_hub to use `load_from_hf_hub`. "
        "See https://pypi.org/project/huggingface-hub/ for installation."
    ) from exc

logger = logging.getLogger(__name__)


def save_checkpoint(cfg: CheckpointConfig, trainer, epoch_itr, val_loss):
    """Save a checkpoint file"""

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return

    trainer.consolidate_optimizer()

    if not trainer.should_save_checkpoint_on_current_rank:
        if trainer.always_call_state_dict_during_save_checkpoint:
            trainer.state_dict()
        return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info("Preparing to save checkpoint for epoch %s @ %s updates", epoch, updates)

    def is_better(metric1, metric2):
        """Check if metric a is better than b."""
        return metric1 >= metric2 if cfg.maximize_best_checkpoint_metric else metric1 <= metric2

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds[f"checkpoint{epoch}{suffix}.pt"] = (
        end_of_epoch and not cfg.no_epoch_checkpoints and epoch % cfg.save_interval == 0
    )
    checkpoint_conds[f"checkpoint_{epoch}_{updates}{suffix}.pt"] = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds[f"checkpoint_best{suffix}.pt"] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and cfg.keep_best_checkpoints > 0:
        worst_best = getattr(save_checkpoint, "best", None)
        chkpts = checkpoint_paths(
            cfg.save_dir,
            pattern=fr"checkpoint\.best_{cfg.best_checkpoint_metric}_(\d+\.?\d*){suffix}\.pt",
        )
        if len(chkpts) > 0:
            last_point = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(last_point.rsplit("_")[-1].replace(f"{suffix}.pt", ""))
        # add random digits to resolve ties
        with data_utils.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            f"checkpoint.best_{cfg.best_checkpoint_metric}_{val_loss:.3f}{rand_sfx}{suffix}.pt"
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        f"checkpoint_last{suffix}.pt"
    ] = not cfg.no_last_checkpoints

    if hasattr(trainer.model, 'pm'):
        extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss,
                        "pruning_manager": trainer.model.pm}
    else:
        extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0 and trainer.should_save_checkpoint_on_current_rank:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for checkpoint in checkpoints[1:]:
            if cfg.write_checkpoints_asynchronously:
                # file copying/moving feature.
                logger.warning(
                    "ioPath is not copying %s to %s since async write mode is on.",
                    checkpoints[0], checkpoint
                )
            else:
                assert PathManager.copy(
                    checkpoints[0], checkpoint, overwrite=True
                ), f"Failed to copy {checkpoints[0]} to {checkpoint}"

        write_timer.stop()
        logger.info(
            "Saved checkpoint %s (epoch %d @ %d updates, score %f) (writing took %f seconds)",
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
        )

    if not end_of_epoch and cfg.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        if cfg.keep_interval_updates_pattern == -1:
            checkpoints = checkpoint_paths(
                cfg.save_dir, pattern=fr"checkpoint_\d+_(\d+){suffix}\.pt"
            )
        else:
            checkpoints = checkpoint_paths(
                cfg.save_dir,
                pattern=fr"checkpoint_\d+_(\d+){suffix}\.pt",
                keep_match=True,
            )
            checkpoints = [
                x[0]
                for x in checkpoints
                if x[1] % cfg.keep_interval_updates_pattern != 0
            ]

        for old_chk in checkpoints[cfg.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            cfg.save_dir, pattern=fr"checkpoint(\d+){suffix}\.pt"
        )
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            cfg.save_dir,
            pattern=fr"checkpoint\.best_{cfg.best_checkpoint_metric}_(\d+\.?\d*){suffix}\.pt",
        )
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)


def load_checkpoint(cfg: CheckpointConfig, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    reset_optimizer = cfg.reset_optimizer
    reset_lr_scheduler = cfg.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(cfg.optimizer_overrides)
    reset_meters = cfg.reset_meters
    reset_dataloader = cfg.reset_dataloader

    if cfg.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )

    suffix = trainer.checkpoint_suffix
    if (
        cfg.restore_file == "checkpoint_last.pt"
    ):  # default value of restore_file is 'checkpoint_last.pt'
        checkpoint_path = os.path.join(
            cfg.save_dir, f"checkpoint_last{suffix}.pt"
        )
        first_launch = not PathManager.exists(checkpoint_path)
        if first_launch and getattr(cfg, "continue_once", None) is not None:
            checkpoint_path = cfg.continue_once
        elif cfg.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint,
            # e.g. restart from last checkpoint and etc.
            if PathManager.exists(cfg.finetune_from_model):
                checkpoint_path = cfg.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    "loading pretrained model from %s: "
                    "optimizer, lr scheduler, meters, dataloader will be reset",
                    checkpoint_path
                )
            else:
                raise ValueError(
                    f"--finetune-from-model {cfg.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = cfg.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = cfg.restore_file

    if cfg.restore_file != "checkpoint_last.pt" and cfg.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def set_param(_model, _name, new_param):
    """Set parameter"""
    _attrs = _name.split('.')
    _parent = _model
    for _attr in _attrs[:-1]:
        _parent = getattr(_parent, _attr)
    setattr(_parent, _attrs[-1], new_param)

def get_param(_model, _name):
    """Get parameter"""
    _attrs = _name.split('.')
    _parent = _model
    for _attr in _attrs[:-1]:
        _parent = getattr(_parent, _attr)
    return getattr(_parent, _attrs[-1])

def load_srp(filename, model):
    """
    Initialize SRP model weights
    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    load_model_state = load_checkpoint_to_cpu(
        filename, load_on_all_ranks=False
        )["model"]
    _dev= None
    for _n, _p in model.named_parameters():
        _c_param = load_model_state[_n]
        if 'embed_tokens' in _n:
            set_param(model, _n, nn.Parameter(_c_param.data))
            if 'decoder.embed_tokens' in _n:
                model.decoder.output_projection.weight = model.decoder.embed_tokens.weight
            continue
        # elif 'output_projection' in _n:
        #     continue

        if not _dev:
            _dev = model.state_dict()[_n].device

        with torch.no_grad():
            if "_indices" in _n or 'mask' in _n:
                set_param(model, _n, nn.Parameter(_c_param.data, requires_grad=False))
            else:
                set_param(model, _n, nn.Parameter(_c_param.data, requires_grad=True))

            if 'fc' in _n and 'weight' in _n:
                _out, _in = _c_param.shape
                _fc = get_param(model, _n[:-7])
                _fc.in_features = _in
                _fc.out_features = _out
                _fc.in_features = _in
                _fc.out_features = _out
    model.to(_dev)
    return model

def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid errors from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_on_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stale for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            os.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)

    with open(local_path, "rb") as file:
        state = torch.load(file, map_location=torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    if "cfg" in state and state["cfg"] is not None:

        # hack to be able to set Namespace in dict config. \
        # this should be removed when we update to newer
        # omegaconf version that supports object flags, or when we migrate all existing models

        old_primitive = _utils.is_primitive_type
        _utils.is_primitive_type = lambda _: True

        state["cfg"] = OmegaConf.create(state["cfg"])

        _utils.is_primitive_type = old_primitive
        OmegaConf.set_struct(state["cfg"], True)

        if arg_overrides is not None:
            overwrite_args_by_name(state["cfg"], arg_overrides)

    state = _upgrade_state_dict(state)
    return state


def load_model_ensemble(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """
    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble, args, _task = load_model_ensemble_and_task(
        filenames,
        arg_overrides,
        task,
        strict,
        suffix,
        num_shards,
        state,
    )
    return ensemble, args


def get_maybe_sharded_checkpoint_filename(
    filename: str, suffix: str, shard_idx: int, num_shards: int
) -> str:
    """Return a sharded checkpoint filename, if it exists."""
    orig_filename = filename
    filename = filename.replace(".pt", suffix + ".pt")
    fsdp_filename = filename[:-3] + f"-shard{shard_idx}.pt"
    model_parallel_filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
    if PathManager.exists(fsdp_filename):
        return fsdp_filename
    elif num_shards > 1:
        return model_parallel_filename
    else:
        return filename


def load_model_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    """Loads an ensemble of models."""
    assert state is None or len(filenames) == 1

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None
    for filename in filenames:
        orig_filename = filename
        model_shard_state = {"shard_weights": [], "shard_metadata": []}
        assert num_shards > 0
        start_time = time.time()
        for shard_idx in range(num_shards):
            filename = get_maybe_sharded_checkpoint_filename(
                orig_filename, suffix, shard_idx, num_shards
            )

            if not PathManager.exists(filename):
                raise IOError(f"Model file not found: {filename}")
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides)
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"Neither args nor cfg exist in state keys = {state.keys()}"
                )

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            if "fsdp_metadata" in state and num_shards > 1:
                model_shard_state["shard_weights"].append(state["model"])
                model_shard_state["shard_metadata"].append(state["fsdp_metadata"])
                # check FSDP import before the code goes too far
                if not has_FSDP:
                    raise ImportError(
                        "Cannot find FullyShardedDataParallel. "
                        "Please install fairscale with: pip install fairscale"
                    )
                if shard_idx == num_shards - 1:
                    consolidated_model_state = FSDP.consolidate_shard_weights(
                        shard_weights=model_shard_state["shard_weights"],
                        shard_metadata=model_shard_state["shard_metadata"],
                    )
                    model = task.build_model(cfg.model)
                    if (
                        "optimizer_history" in state
                        and len(state["optimizer_history"]) > 0
                        and "num_updates" in state["optimizer_history"][-1]
                    ):
                        model.set_num_updates(
                            state["optimizer_history"][-1]["num_updates"]
                        )
                    model = load_srp(filename, model)
                    model.load_state_dict(
                        consolidated_model_state, strict=strict, model_cfg=cfg.model
                    )
            else:
                # model parallel checkpoint or unsharded checkpoint
                # support old external tasks

                argspec = inspect.getfullargspec(task.build_model)
                if "from_checkpoint" in argspec.args:
                    model = task.build_model(cfg.model, from_checkpoint=True)
                else:
                    model = task.build_model(cfg.model)
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    model.set_num_updates(state["optimizer_history"][-1]["num_updates"])
                model = load_srp(filename, model)
                model.load_state_dict(
                    state["model"], strict=strict, model_cfg=cfg.model
                )

            if 'extra_state' in state:
                if 'pruning_manager' in state['extra_state']:
                    model.pm = state['extra_state']['pruning_manager']

            # reset state so it gets loaded for the next model in ensemble
            state = None
            if shard_idx % 10 == 0 and shard_idx > 0:
                elapsed = time.time() - start_time
                logger.info(
                "Loaded %f shards in %.2fs, %.2fs/shard",
                shard_idx, elapsed, elapsed / (shard_idx+1)
                )

        # build model for ensemble
        ensemble.append(model)
    return ensemble, cfg, task


def load_model_ensemble_and_task_from_hf_hub(
    model_id,
    cache_dir: Optional[str] = None,
    arg_overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """Loads an ensemble of models from the huggingface hub."""

    library_name = "fairseq"
    cache_dir = cache_dir or (Path.home() / ".cache" / library_name).as_posix()
    cache_dir = snapshot_download(
        model_id, cache_dir=cache_dir, library_name=library_name, **kwargs
    )

    _arg_overrides = arg_overrides or {}
    _arg_overrides["data"] = cache_dir
    return load_model_ensemble_and_task(
        [p.as_posix() for p in Path(cache_dir).glob("*.pt")],
        arg_overrides=_arg_overrides,
    )


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt", keep_match=False):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = PathManager.ls(path)

    entries = []
    for i, file in enumerate(files):
        match = pt_regexp.fullmatch(file)
        if match is not None:
            idx = float(match.group(1)) if len(match.groups()) > 0 else i
            entries.append((idx, match.group(0)))
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse=True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(obj, filename, async_write: bool = False):
    """Saves a torch object to a file."""
    if async_write:
        with PathManager.opena(filename, "wb") as file:
            _torch_persistent_save(obj, file)
    else:
        if PathManager.supports_rename(filename):
            # do atomic save
            with PathManager.open(filename + ".tmp", "wb") as file:
                _torch_persistent_save(obj, file)
            PathManager.rename(filename + ".tmp", filename)
        else:
            # fallback to non-atomic save
            with PathManager.open(filename, "wb") as file:
                _torch_persistent_save(obj, file)


def _torch_persistent_save(obj, file):
    """Saves a torch object to a file."""
    if isinstance(file, str):
        with PathManager.open(file, "wb") as sub_file:
            torch_persistent_save(obj, sub_file)
        return None
    for i in range(3):
        try:
            return torch.save(obj, file)
        except OSError:
            if i == 2:
                logger.error(traceback.format_exc())
                raise
            return None

def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"].get("epoch", 0),
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }

    # backward compatibility, cfg updates
    if "args" in state and state["args"] is not None:
        # old model checkpoints may not have separate source/target positions
        if hasattr(state["args"], "max_positions") and not hasattr(
            state["args"], "max_source_positions"
        ):
            state["args"].max_source_positions = state["args"].max_positions
            state["args"].max_target_positions = state["args"].max_positions
        # default to translation task
        if not hasattr(state["args"], "task"):
            state["args"].task = "translation"
        # --raw-text and --lazy-load are deprecated
        if getattr(state["args"], "raw_text", False):
            state["args"].dataset_impl = "raw"
        elif getattr(state["args"], "lazy_load", False):
            state["args"].dataset_impl = "lazy"
        # epochs start at 1
        if state["extra_state"]["train_iterator"] is not None:
            state["extra_state"]["train_iterator"]["epoch"] = max(
                state["extra_state"]["train_iterator"].get("epoch", 1), 1
            )
        # --remove-bpe ==> --postprocess
        if hasattr(state["args"], "remove_bpe"):
            state["args"].post_process = state["args"].remove_bpe
        # --min-lr ==> --stop-min-lr
        if hasattr(state["args"], "min_lr"):
            state["args"].stop_min_lr = state["args"].min_lr
            del state["args"].min_lr
        # binary_cross_entropy / kd_binary_cross_entropy => wav2vec criterion
        if hasattr(state["args"], "criterion") and state["args"].criterion in [
            "binary_cross_entropy",
            "kd_binary_cross_entropy",
        ]:
            state["args"].criterion = "wav2vec"
        # remove log_keys if it's None (criteria will supply a default value of [])
        if hasattr(state["args"], "log_keys") and state["args"].log_keys is None:
            delattr(state["args"], "log_keys")
        # speech_pretraining => audio pretraining
        if (
            hasattr(state["args"], "task")
            and state["args"].task == "speech_pretraining"
        ):
            state["args"].task = "audio_pretraining"
        # audio_cpc => wav2vec
        if hasattr(state["args"], "arch") and state["args"].arch == "audio_cpc":
            state["args"].arch = "wav2vec"
        # convert legacy float learning rate to List[float]
        if hasattr(state["args"], "lr") and isinstance(state["args"].lr, float):
            state["args"].lr = [state["args"].lr]
        # convert task data arg to a string instead of List[string]
        if (
            hasattr(state["args"], "data")
            and isinstance(state["args"].data, list)
            and len(state["args"].data) > 0
        ):
            state["args"].data = state["args"].data[0]

        state["cfg"] = convert_namespace_to_omegaconf(state["args"])

    if "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
        with open_dict(cfg):
            # any upgrades for Hydra-based configs
            if (
                "task" in cfg
                and "eval_wer_config" in cfg.task
                and isinstance(cfg.task.eval_wer_config.print_alignment, bool)
            ):
                cfg.task.eval_wer_config.print_alignment = "hard"
            if "generation" in cfg and isinstance(cfg.generation.print_alignment, bool):
                cfg.generation.print_alignment = (
                    "hard" if cfg.generation.print_alignment else None
                )
            if (
                "model" in cfg
                and "w2v_args" in cfg.model
                and cfg.model.w2v_args is not None
                and (
                    hasattr(cfg.model.w2v_args, "task") or "task" in cfg.model.w2v_args
                )
                and hasattr(cfg.model.w2v_args.task, "eval_wer_config")
                and cfg.model.w2v_args.task.eval_wer_config is not None
                and isinstance(
                    cfg.model.w2v_args.task.eval_wer_config.print_alignment, bool
                )
            ):
                cfg.model.w2v_args.task.eval_wer_config.print_alignment = "hard"

    return state


def prune_state_dict(state_dict, model_cfg: Optional[DictConfig]):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - "
        "this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            int(layer_string) for layer_string in layers_to_keep.split(",")
        )
        mapping_dict = {}
        for i, module in enumerate(keep_layers):
            mapping_dict[str(module)] = str(i)

        regex = re.compile(fr"^{layer_name}.*\.layers\.(\d+)")
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if isinstance(model_cfg, DictConfig):
        context = open_dict(model_cfg)
    else:
        context = contextlib.ExitStack()
    with context:
        if hasattr(model_cfg, "encoder_layers_to_keep"):
            model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"):
            model_cfg.decoder_layers_to_keep = None

    return new_state_dict


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder],
    checkpoint: str,
    strict: bool = True,
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError(f"Model file not found: {checkpoint}")
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=strict)
    return component


def verify_checkpoint_directory(save_dir: str) -> None:
    """Verify that the checkpoint directory exists."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w", encoding="utf-8"):
            pass
    except OSError as error:
        logger.warning(
            "Unable to access checkpoint save directory: %s", save_dir
        )
        raise error
    else:
        os.remove(temp_file_path)


def save_ema_as_checkpoint(src_path, dst_path):
    """Saves exponential moving averaged (EMA) checkpoint from input."""
    state = load_ema_from_checkpoint(src_path)
    torch_persistent_save(state, dst_path)

def load_ema_from_checkpoint(fpath):
    """Loads exponential moving averaged (EMA) checkpoint from input and
    returns a model with ema weights.

    Args:
      fpath: A string path of checkpoint to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    new_state = None

    with PathManager.open(fpath, "rb") as file:
        new_state = torch.load(
            file,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

        # EMA model is stored in a separate "extra state"
        model_params = new_state["extra_state"]["ema"]

        for key in list(model_params.keys()):
            parma = model_params[key]
            if isinstance(parma, torch.HalfTensor):
                parma = parma.float()
            if key not in params_dict:
                params_dict[key] = parma.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                raise ValueError(f"Key {key} is repeated in EMA model params.")

        if len(params_dict) == 0:
            raise ValueError(
                f"Input checkpoint path '{fpath}' does not contain "
                "ema model weights, is this model trained with EMA?"
            )

    new_state["model"] = params_dict
    return new_state
