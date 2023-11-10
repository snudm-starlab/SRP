#!/usr/bin/env python3 -u
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
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from src import checkpoint_utils
from src.flops_counter import FlopsCounter


def main(cfg: DictConfig):
    """Main function for generation."""

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            f"generate-{cfg.dataset.gen_subset}.txt",
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as file:
            return _main(cfg, file)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    """Return a set of symbols to strip from the output."""
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from %s", cfg.common_eval.path)
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded
    # so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                "Failed to load language model! Please make sure that the language model dict"
                "is the same as target dict and is located in the data dir %s", cfg.task.data
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(input_str):
        if bpe is not None:
            input_str = bpe.decode(input_str)
        if tokenizer is not None:
            input_str = tokenizer.decode(input_str)
        return input_str

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print(f"S-{sample_id}\t{src_str}", file=output_file)
                if has_target:
                    print(f"T-{sample_id}\t{target_str}", file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        f"H-{sample_id}\t{score}\t{hypo_str}",
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        f"D-{sample_id}\t{score}\t{detok_hypo_str}",
                        file=output_file,
                    )
                    # convert from base e to base 2
                    _tmp =[f'{x:.4f}' for x in hypo['positional_scores'].div_(math.log(2)).tolist()]
                    _tmp = ' '.join(_tmp)
                    print(
                        f"P-{sample_id}\t{_tmp}",
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        _tmp = ' '.join([f'{src_idx}-{tgt_idx}' for src_idx, tgt_idx in alignment])
                        print(
                            f"A-{sample_id}\t{_tmp}",
                                    file=output_file)

                    if cfg.generation.print_alignment == "soft":
                        _tmp = " ".join([",".join(src_probs) for src_probs in alignment])
                        print(f"A-{sample_id}\t{_tmp}", file=output_file)

                    if cfg.generation.print_step:
                        print(f"I-{sample_id}\t{hypo['steps']}", file=output_file)

                    if cfg.generation.retain_iter_history:
                        for step, his in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=his["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(f"E-{sample_id}_{step}\t{h_str}", file=output_file)

                # Score only the top hypothesis
                if has_target and j == 0:
                    if (
                        align_dict is not None
                        or cfg.common_eval.post_process is not None
                    ):
                        # Convert back to tokens for evaluation w/ unk replacement and/or w/o BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Translated %d sentences (%d tokens) in %.1f}s (%.2f sentences/s, %.2f tokens/s)",
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences/gen_timer.sum, 1./gen_timer.avg)

    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, "
                    "this is probably not what you want. "
                    "Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, "
                    "the BLEU score is computed on BPE tokens, "
                    "not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            f"Generate {cfg.dataset.gen_subset} with beam={cfg.generation.beam}:"
            f"{scorer.result_string()}",
            file=output_file,
        )

    # print the number of parameters and FLOPs
    num_params = np.sum([_p.numel() for _p in models[0].parameters() if _p.requires_grad])
    param_dict = models[0].state_dict()
    seq_len = 50
    heads = 4
    num_layers = 6
    emb = param_dict['encoder.embedding_c'].shape[0]
    en_self_qks = [param_dict[f'encoder.layers.{l}.self_attn_qk_c'].shape[0]
            for l in range(num_layers)]
    en_self_vos = [param_dict[f'encoder.layers.{l}.self_attn_vo_c'].shape[0]
            for l in range(num_layers)]
    en_fcs = [param_dict[f'encoder.layers.{l}.fc_c'].shape[0] for l in range(num_layers)]

    de_self_qks = [param_dict[f'decoder.layers.{l}.self_attn_qk_c'].shape[0] \
            for l in range(num_layers)]
    de_self_vos = [param_dict[f'decoder.layers.{l}.self_attn_vo_c'].shape[0] \
            for l in range(num_layers)]
    de_encoder_qks = [param_dict[f'decoder.layers.{l}.encoder_attn_qk_c'].shape[0] \
            for l in range(num_layers)]
    de_encoder_vos = [param_dict[f'decoder.layers.{l}.encoder_attn_vo_c'].shape[0] \
            for l in range(num_layers)]
    de_fcs = [param_dict[f'decoder.layers.{l}.fc_c'].shape[0] for l in range(num_layers)]

    tar_dict_size = 6632

    flops = FlopsCounter(seq_len, emb, heads,
                en_self_qks, en_self_vos, en_fcs,
                de_self_qks, de_self_vos, de_fcs,
                de_encoder_qks, de_encoder_vos,
                tar_dict_size).get_model_flops()

    print(f"* Number of params: {num_params/1e6:.3f}M")
    print(f"* FLOPs: {flops/1e9:.3f}G")
    print(f"* {scorer.result_string().split(',')[0]}")
    return scorer


def cli_main():
    """Main function."""
    parser = options.get_generation_parser()
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
