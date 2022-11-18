# Ref: https://github.com/google-research/electra/blob/master/flops_computation.py

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 8


class FLOPS_COUNTER:
    def __init__(self, s, emb, heads,
                       en_self_qks, en_self_vos, en_fcs,
                       de_self_qks, de_self_vos, de_fcs,
                       de_encoder_qks, de_encoder_vos,
                       tar_dict_size,
                        ):
        self.s = s # sequence length
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

    # FLOPs for sub-layers
    def get_attn_flops(self, qk, vo, _s, compute_kv=True):
        # emb, qk, vo: dimensions
        attn_flops = dict(
            # projection
            q_proj = 2 * self.emb * qk,
            q_bias = qk,

            # attention
            attn_scores = 2 * qk * _s,
            attn_softmax = SOFTMAX_FLOPS * _s * self.heads,
            attn_dropout = DROPOUT_FLOPS * _s * self.heads,
            attn_scale = _s * self.heads,
            attn_weight_avg_values = 2 * self.emb * _s,
            # attn_output projection
            attn_output = 2 * self.emb * self.emb,
            attn_output_bias = self.emb,
            attn_output_dropout = DROPOUT_FLOPS * self.emb,
            # residual connection
            residual = self.emb,
            # Layer norm
            layer_norm = LAYER_NORM_FLOPS
        )
        kv_flops = dict(
            k_proj = 2 * self.emb * qk,
            k_bias = qk,
            v_proj = 2 * self.emb * vo,
            v_bias = vo,
            )
        if compute_kv:
            attn_flops.update(kv_flops)
        return sum(attn_flops.values()) * _s

    def get_fc_flops(self, fc, _s):
        # fc: intermediate dimension
        fc_flops = dict(
            # first fc layer
            intermediate = 2 * self.emb * fc,
            intermediate_act = ACTIVATION_FLOPS * fc,
            intermediate_bias = fc,
            # second fc layer
            output = 2 * fc * self.emb,
            output_bias = self.emb,
            output_dropout = DROPOUT_FLOPS * self.emb,
            # residual
            output_residual = self.emb,
            # layernrom
            output_layer_norm = LAYER_NORM_FLOPS * self.emb,
        )

        return sum(fc_flops.values()) * _s

    # FLOPs for a single layer
    def get_layer_flops(self, self_qk=None, self_vo=None, 
                        encoder_qk=None, encoder_vo=None, 
                        fc=None, sequence_length=-1):
        if sequence_length == -1:
            sequence_length = self.s

        layer_flops = {}
        layer_flops['self_attn'] = self.get_attn_flops(
                                    self_qk, self_vo, sequence_length)
        if encoder_qk and encoder_vo:                            
            layer_flops['encoder_attn'] = self.get_attn_flops(
                                        self_qk, self_vo, sequence_length)
        layer_flops['fc'] = self.get_fc_flops(fc, sequence_length)
        return sum(layer_flops.values())

    def get_encoder_flops(self, self_qks, self_vos, fcs):
        _tot = 0
        for i in range(len(self_qks)):
            # Accumulate FLOPs of each layer
            _tot += self.get_layer_flops(self_qk=self_qks[i],
                                         self_vo=self_vos[i],
                                         fc=fcs[i])
        return _tot

    def get_decoder_flops(self, self_qks, self_vos, 
                                encoder_qks, encoder_vos, fcs,
                                sequence_length):
        _tot = 0
        for sl in range(1, sequence_length+1):
            for i in range(len(self_qks)):
                _tot += self.get_layer_flops(self_qk=self_qks[i],
                                            self_vo=self_vos[i],
                                            encoder_qk=encoder_qks[i],
                                            encoder_vo=encoder_vos[i],
                                            fc=fcs[i],
                                            sequence_length=sequence_length
                                            )
        return _tot

    def get_classification_flops(self,):
        classification_flops = dict(
            hidden = 2 * self.emb * self.tar_dict_size,
            hidden_bias = self.tar_dict_size,
            )
        return sum(classification_flops.values()) * self.s

    def get_model_flops(self,):
        en_flops = self.get_encoder_flops(self.en_self_qks,
                                          self.en_self_vos,
                                          self.en_fcs)
        de_flops = self.get_decoder_flops(self.de_self_qks,
                                         self.de_self_vos,
                                         self.de_encoder_qks,
                                         self.de_encoder_vos,
                                         self.de_fcs,
                                         sequence_length= self.s)
        cls_flops = self.get_classification_flops()
        return en_flops + de_flops + cls_flops

if __name__ == "__main__":
    s = 50

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
    tar_dict_size = 6632
                
    fc1 = FLOPS_COUNTER(s, emb, heads,
                en_self_qks, en_self_vos, en_fcs,
                de_self_qks, de_self_vos, de_fcs,
                de_encoder_qks, de_encoder_vos,
                tar_dict_size)
    flops1 = fc1.get_model_flops()
    FLOPs1.append(flops1)
    f1 = sum(FLOPs1) / 1e9
    print(f1)
    ######################################################
    

    ############## COMPRESSED MODEL ######################
    FLOPs2 = []
    emb, heads = 259, 4
    en_self_qks = [256, 256, 256]
    en_self_vos = [256, 256, 256]
    en_fcs = [451, 476, 531]


    de_self_qks = [256] * 3
    de_self_vos = [256] * 3
    de_encoder_qks = [256] * 3
    de_encoder_vos = [256] * 3
    de_fcs = [491, 552, 607]
    tar_dict_size = 6632

    """
    FLOPs2 = []
    emb, heads = 198, 4
    en_self_qks = [156,236,224,132,160,156]
    en_self_vos = [268,272,248,180,192,184]
    en_fcs = [324,333,311,340,293,484]


    de_self_qks = [372,188,196,196,244,356]
    de_self_vos = [320,208,168,180,192,196]
    de_encoder_qks = [236,224,128,152,124,104]
    de_encoder_vos = [268,236,140,188,92,52]
    de_fcs = [335,299,446,473,481,566]
    tar_dict_size = 6632
    """
                
    fc2 = FLOPS_COUNTER(s, emb, heads,
                en_self_qks, en_self_vos, en_fcs,
                de_self_qks, de_self_vos, de_fcs,
                de_encoder_qks, de_encoder_vos,
                tar_dict_size)
    flops2 = fc2.get_model_flops()
    FLOPs2.append(flops2)
    f2 = sum(FLOPs2) / 1e9
    ####################################################

    # Print
    print("* FLOPs")
    print(f"- Base model: {f1:.2f}G")
    print(f"- Comp. model: {f2:.2f}G")
    print(f"- Comp. rate: {f2/f1*100:.2f}%")

