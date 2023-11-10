"""
Starlab Transformer Compression with SRP (Sparse Regularization Pruning)

unit_test.py
- Check if the function of SRP works well
"""

import unittest
import copy
import numpy as np

import torch

import fairseq
from fairseq import options, tasks, utils
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

import checkpoint_utils

class SRPTest(unittest.TestCase):
    """ A Class for unittest of SRP"""
    def tearDown(self):
        """ This function is called after each test"""
        print("Test Case Complete!!!")
    # Encoder Test
    def test1_srp_encoder_shape(self): # MHA
        """Check the output dimension of SRP Encoder"""
        self.assertEqual(srp_encoder_out.shape, encoder_out.shape)

    def test2_compare_encoder_params(self): # Transformer layer
        """Check srp encoder is larger than pruned srp encoder """
        self.assertTrue(srp_en_params > pruned_en_params)

    def test3_pruned_srp_encoder_shape(self):
        """Check the output dimension of Pruned SRP Encoder"""
        self.assertEqual(srp_encoder_out.shape, pruned_srp_encoder_out.shape)

    # Decoder Test
    def test4_srp_decoder_shape(self):
        """Check the output dimension of SRP Decoder"""
        self.assertEqual(srp_decoder_out[0].shape, decoder_out[0].shape)

    def test5_compare_decoder_params(self):
        """Check srp decoder is larger than pruned srp decoder """
        self.assertTrue(srp_de_params > pruned_de_params)

    def test6_pruned_srp_decoder_shape(self):
        """Check the output dimension of Pruned SRP Decoder"""
        self.assertEqual(srp_decoder_out[0].shape, pruned_srp_decoder_out[0].shape)


if __name__ == "__main__":
    parser = options.get_training_parser()
    input_args = ["../data-bin/iwslt14.tokenized.de-en",
                  "--user-dir", "../src",
                  "--arch", "srp_iwslt_de_en",
                  "--share-decoder-input-output-embed",
                  "--task", "SRPtranslation",
                  ]
    args = options.parse_args_and_arch(parser, input_args = input_args)
    CFG = convert_namespace_to_omegaconf(args)

    utils.import_user_module(CFG.common)
    add_defaults(CFG)

    np.random.seed(CFG.common.seed)
    utils.set_torch_seed(CFG.common.seed)

    task = tasks.setup_task(CFG.task)
    model = task.build_model(CFG.model)

    # For fairseq Encoder & Decoder
    CFG.encoder_ffn_embed_dim = 1024
    CFG.encoder_attention_heads = 4
    CFG.decoder_ffn_embed_dim = 1024
    CFG.decoder_attention_heads = 4

    encoder = fairseq.modules.TransformerEncoderLayer(CFG)
    decoder = fairseq.modules.TransformerDecoderLayer(CFG)

    en_params = sum(p.numel() for p in encoder.parameters())
    de_params = sum(p.numel() for p in decoder.parameters())

    srp_encoder = model.encoder.layers[0]
    srp_decoder = model.decoder.layers[0]

    srp_en_params = sum(p.numel() for p in srp_encoder.parameters())
    srp_de_params = sum(p.numel() for p in srp_decoder.parameters())

    # Prepare data
    dummy_tokens = torch.rand(25, 1, 512)# batch_size * src_len
    dummy_decoder_input = torch.rand(25, 1, 512 )  # tgt_len, batch_size, embed_dim

    encoder_out = encoder(dummy_tokens, encoder_padding_mask=None)
    decoder_out = decoder(dummy_decoder_input, encoder_out=encoder_out)

    srp_encoder_out = srp_encoder(dummy_tokens, encoder_padding_mask=None)
    srp_decoder_out = srp_decoder(dummy_decoder_input, encoder_out=srp_encoder_out)

    # Pruned Model
    state_dict = torch.load("../checkpoints/stage2/checkpoint_best.pt")
    pruned_model = copy.deepcopy(model)
    pruned_model = checkpoint_utils.load_srp("../checkpoints/stage2/checkpoint_best.pt",
                                             pruned_model)
    pruned_srp_encoder = pruned_model.encoder.layers[0]
    pruned_srp_decoder = pruned_model.decoder.layers[0]

    pruned_en_params = sum(p.numel() for p in pruned_srp_encoder.parameters())
    pruned_de_params = sum(p.numel() for p in pruned_srp_decoder.parameters())

    pruned_srp_encoder_out = pruned_srp_encoder(dummy_tokens, encoder_padding_mask=None)
    pruned_srp_decoder_out = pruned_srp_decoder(dummy_decoder_input,
                                                encoder_out=pruned_srp_encoder_out)

    unittest.main()
