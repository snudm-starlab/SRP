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
# [GitHub Repository]: https://github.com/google-research/electra/blob/master/flops_computation.py
################################################################################


"""
Counting FLOPs of the Transformer model.
"""

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 8


class FlopsCounter:
    """
    class for counting the numer of FLOP
    for generating sequence of the given length
    """
    def __init__(self, seq_len, emb, heads,
                       en_self_qks, en_self_vos, en_fcs,
                       de_self_qks, de_self_vos, de_fcs,
                       de_encoder_qks, de_encoder_vos,
                       tar_dict_size,
                        ):
        self.seq_len = seq_len # sequence length
        self.emb = emb # embedding dim
        self.heads = heads # attention heads

        self.en_self_qks = en_self_qks
        self.en_self_vos = en_self_vos
        self.en_fcs = en_fcs

        self.de_self_qks = de_self_qks
        self.de_self_vos = de_self_vos
        self.de_encoder_qks = de_encoder_qks
        self.de_encoder_vos = de_encoder_vos
        self.de_fcs = de_fcs

        self.tar_dict_size = tar_dict_size

    def get_attn_flops(self, qk_dim, vo_dim, _s, compute_kv=True):
        """compute the FLOPs of the attention sub-layer"""
        # emb, qk, vo: dimensions
        attn_flops = {
            # projection
            "q_proj": 2 * self.emb * qk_dim,
            "q_bias": qk_dim,

            # attention
            "attn_scores": 2 * qk_dim * _s,
            "attn_softmax": SOFTMAX_FLOPS * _s * self.heads,
            "attn_dropout": DROPOUT_FLOPS * _s * self.heads,
            "attn_scale": _s * self.heads,
            "attn_weight_avg_values": 2 * self.emb * _s,
            # attn_output projection
            "attn_output": 2 * self.emb * self.emb,
            "attn_output_bias": self.emb,
            "attn_output_dropout": DROPOUT_FLOPS * self.emb,
            # residual connection
            "residual": self.emb,
            # Layer norm
            "layer_norm": LAYER_NORM_FLOPS
        }
        kv_flops = {
            "k_proj": 2 * self.emb * qk_dim,
            "k_bias": qk_dim,
            "v_proj": 2 * self.emb * vo_dim,
            "v_bias": vo_dim,
        }
        if compute_kv:
            attn_flops.update(kv_flops)
        return sum(attn_flops.values()) * _s

    def get_fc_flops(self, fc_layer, _s):
        """compute the FLOPs of the feedforward sub-layer"""
        fc_flops = {
            # first fc layer
            "intermediate": 2 * self.emb * fc_layer,
            "intermediate_act": ACTIVATION_FLOPS * fc_layer,
            "intermediate_bias": fc_layer,
            # second fc layer
            "output": 2 * fc_layer * self.emb,
            "output_bias": self.emb,
            "output_dropout": DROPOUT_FLOPS * self.emb,
            # residual
            "output_residual": self.emb,
            # layernrom
            "output_layer_norm": LAYER_NORM_FLOPS * self.emb,
        }

        return sum(fc_flops.values()) * _s

    def get_layer_flops(self, self_qk=None, self_vo=None,
                        encoder_qk=None, encoder_vo=None,
                        fc_layer=None, sequence_length=-1):
        """Compute the FLOPs of a single layer"""
        if sequence_length == -1:
            sequence_length = self.seq_len

        layer_flops = {}
        layer_flops['self_attn'] = self.get_attn_flops(
                                    self_qk, self_vo, sequence_length)
        if encoder_qk and encoder_vo:
            layer_flops['encoder_attn'] = self.get_attn_flops(
                                        self_qk, self_vo, sequence_length)
        layer_flops['fc'] = self.get_fc_flops(fc_layer, sequence_length)
        return sum(layer_flops.values())

    def get_encoder_flops(self, self_qks, self_vos, fcs):
        """Compute the FLOPs of an encoder"""
        _tot = 0
        for idx, self_qk in enumerate(self_qks):
            # Accumulate FLOPs of each layer
            _tot += self.get_layer_flops(self_qk=self_qk,
                                         self_vo=self_vos[idx],
                                         fc_layer=fcs[idx])
        return _tot

    def get_decoder_flops(self, self_qks, self_vos,
                                encoder_qks, encoder_vos, fcs,
                                sequence_length):
        """Compute the FLOPs of a decoder"""
        _tot = 0
        for _ in range(1, sequence_length+1):
            for idx, self_qk in enumerate(self_qks):
                _tot += self.get_layer_flops(self_qk=self_qk,
                                            self_vo=self_vos[idx],
                                            encoder_qk=encoder_qks[idx],
                                            encoder_vo=encoder_vos[idx],
                                            fc_layer=fcs[idx],
                                            sequence_length=sequence_length
                                            )
        return _tot

    def get_classification_flops(self,):
        """Compute the FLOPs of the classification layer"""
        classification_flops = {
            "hidden": 2 * self.emb * self.tar_dict_size,
            "hidden_bias": self.tar_dict_size,
        }
        return sum(classification_flops.values()) * self.seq_len

    def get_model_flops(self,):
        """Compute the FLOPs of the whole model"""
        en_flops = self.get_encoder_flops(self.en_self_qks,
                                          self.en_self_vos,
                                          self.en_fcs)
        de_flops = self.get_decoder_flops(self.de_self_qks,
                                         self.de_self_vos,
                                         self.de_encoder_qks,
                                         self.de_encoder_vos,
                                         self.de_fcs,
                                         sequence_length= self.seq_len)
        cls_flops = self.get_classification_flops()
        return en_flops + de_flops + cls_flops

if __name__ == "__main__":
    S = 50

    ################### BASE LINE MODEL ###################
    FLOPs1 = []
    emb, heads = 512, 4
    en_self_qks = [512] * 6
    en_self_vos = [512] * 6
    en_fcs = [1024] * 6


    de_self_qks = [512] * 6
    de_self_vos = [512] * 6
    de_encoder_qks = [512] * 6
    de_encoder_vos = [512] * 6
    de_fcs = [1024] * 6
    TAR_DICT_SIZE = 6632

    fc1 = FlopsCounter(S, emb, heads,
                en_self_qks, en_self_vos, en_fcs,
                de_self_qks, de_self_vos, de_fcs,
                de_encoder_qks, de_encoder_vos,
                TAR_DICT_SIZE)
    flops1 = fc1.get_model_flops()
    FLOPs1.append(flops1)
    f1 = sum(FLOPs1) / 1e9
    ######################################################

    ############## COMPRESSED MODEL ######################
    FLOPs2 = []
    emb, heads, fnn = 80, 4, 160
    en_self_qks = [emb]*6
    en_self_vos = [emb]*6
    en_fcs = [fnn]*6


    de_self_qks = [emb]*6
    de_self_vos = [emb]*6
    de_encoder_qks = [emb]*6
    de_encoder_vos = [emb]*6
    de_fcs = [fnn]*6
    TAR_DICT_SIZE = 6632

    fc2 = FlopsCounter(S, emb, heads,
                en_self_qks, en_self_vos, en_fcs,
                de_self_qks, de_self_vos, de_fcs,
                de_encoder_qks, de_encoder_vos,
                TAR_DICT_SIZE)
    flops2 = fc2.get_model_flops()
    FLOPs2.append(flops2)
    f2 = sum(FLOPs2) / 1e9
    ####################################################

    # Print the results
    print("* FLOPs")
    print(f"- Base model: {f1:.2f}G")
    print(f"- Comp. model: {f2:.2f}G")
    print(f"- Comp. rate: {f2/f1*100:.2f}%")
