"""
Starlab Transformer Compression with SRP (Sparse Regularization Pruning)

unit_test.py
- Check if the function of SRP works well
"""

import pickle
import argparse
import unittest
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from fairseq import options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

class SRPTest(unittest.TestCase):
    """ A Class for unittest of SRP"""

    # Encoder Test
    def test1(self): # MHA
        raise NotImplementedError
    
    def test2(self): # Transformer layer
        raise NotImplementedError
    
    def test3(self):
        raise NotImplementedError
    
    # Decoder Test
    def test4(self):
        raise NotImplementedError
    
    def test5(self):
        raise NotImplementedError
    
    def test6(self):
        raise NotImplementedError
    
    # SRP Test
    def test7(self):
        raise NotImplementedError
    
    def test8(self):
        raise NotImplementedError

    def test9(self):
        raise NotImplementedError

def main(cfg: FairseqConfig) -> None:
    """ Main function for unittest of SRP"""
    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    task = tasks.setup_task(cfg.task)
    # print(f"TASK: {task}\n")

    model = task.build_model(cfg.model)
    # print(f"Model: {model}")

    encoder = model.encoder.layers[0]
    decoder = model.decoder.layers[0]

    # print(f"Encoder Layer: {encoder}")
    # print("\n" + "=" * 30 + "\n")
    # print(f"Decoder: {decoder}")

    en_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    de_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    # print(f"Encoder: {en_params:.2f}M parameters")
    # print(f"Decoder: {de_params:.2f}M parameters")


    # Data Check
    with open('../data-bin/iwslt14.tokenized.de-en/samples/samples0.pkl', 'rb') as file:
        samples = pickle.load(file)

    # Samples
    # for _key, _value in samples[0].items():
        # print(f"Key: {_key}, Type of Value: {type(_value)}")

    net_input = samples[0]['net_input']

    # 'net_input' of samples
    # for _key, _value in net_input.items():
    #     print(f"Key: {_key}, Type of Value: {type(_value)}, Shape of Value: {_value.shape}")
    print(model)
    src_tokens = net_input['src_tokens']
    print(f"Source Tokens: {src_tokens.shape}")
    src_lengths = net_input['src_lengths']
    print(f"Source Lengths: {src_lengths.shape}")

    encoder_out = model.encoder(src_tokens, src_lengths=src_lengths)
    print(encoder_out['encoder_out'][0].shape)
    # for _key, _value in encoder_out.items():
    #     print(f"Key: {_key}, Type of Value: {type(_value)}")

    decoder_input = torch.rand(25, 144, 512 )  # tgt_len, batch_size, embed_dim
    # decoder_input = net_input['prev_output_tokens']
    print(f"Decoder Input: {decoder_input.shape}")
    decoder_out = decoder(decoder_input, encoder_out=encoder_out['encoder_out'][0])
    # print(f"Decoder Output: {decoder_out[0].shape}")  # tgt_len, batch_size, embed_dim
    # print(f"Decoder Output: {type(decoder_out[1])}")  # None
    # print(f"Decoder Output: {type(decoder_out[2])}")  # None
    # print(decoder_out)

def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    """ Main function for unittest of SRP"""
    parser = options.get_training_parser()
    input_args = ["../data-bin/iwslt14.tokenized.de-en", "--user-dir", "../src", "--arch", "srp_iwslt_de_en",
                  "--share-decoder-input-output-embed", "--task", "SRPtranslation"]
    args = options.parse_args_and_arch(parser, input_args = input_args)
    cfg = convert_namespace_to_omegaconf(args)
    main(cfg)

if __name__ == "__main__":
    cli_main()
