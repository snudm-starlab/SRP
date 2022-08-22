SOFTMAX_FLOPS = 5
DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8


########################### Computing FLOPs #####################

def get_flops(
                bs=None,
                kq_dim=None,
                v_dim=None,
                seq_len=None,
                num_heads=None,
                emb_dim=None,
                ffn_dim=None,
                input_bits=None,
                encoder_bits=None,
                decoder_bits=None,
                **kwargs,
            ):
    _flops = 0
    _flops += get_encoder_flops(bs,kq_dim,v_dim,seq_len,num_heads,emb_dim,ffn_dim,
                                input_bits, encoder_bits)
    _flops += get_decoder_flops(bs,kq_dim,v_dim,seq_len,num_heads,emb_dim,ffn_dim,
                                input_bits, decoder_bits)
    return _flops


def get_decoder_flops(
                      bs=None,
                      kq_dim=None,
                      v_dim=None,
                      seq_len=None,
                      num_heads=None,
                      emb_dim=None,
                      ffn_dim=None,
                      input_bits=None,
                      ffn_bits=None,
                      ):
    _flops = 0
    for _bits in ffn_bits:
        _flops += get_decoder_layer_flops(
                                          bs=bs,
                                          kq_dim=kq_dim,
                                          v_dim=v_dim,
                                          seq_len=seq_len,
                                          num_heads=num_heads,
                                          emb_dim=emb_dim,
                                          ffn_dim=ffn_dim,
                                          input_bits=input_bits,
                                          ffn_bits=_bits,
                                          num_attns=2,
                                          )

    return _flops

def get_encoder_flops(
                      bs=None,
                      kq_dim=None,
                      v_dim=None,
                      seq_len=None,
                      num_heads=None,
                      emb_dim=None,
                      ffn_dim=None,
                      input_bits=None,
                      ffn_bits=None,
                      ):
    _flops = 0
    for _bits in ffn_bits:
        _flops += get_encoder_layer_flops(
                                          bs=bs,
                                          kq_dim=kq_dim,
                                          v_dim=v_dim,
                                          seq_len=seq_len,
                                          num_heads=num_heads,
                                          emb_dim=emb_dim,
                                          ffn_dim=ffn_dim,
                                          input_bits=input_bits,
                                          ffn_bits=_bits,
                                          num_attns=1,
                                          )

    return _flops




def get_decoder_layer_flops(
                      bs=None,
                      kq_dim=None,
                      v_dim=None,
                      seq_len=None,
                      num_heads=None,
                      emb_dim=None,
                      ffn_dim=None,
                      input_bits=None,
                      ffn_bits=None,
                      num_attns=2,
                        ):
    assert input_bits in [16, 32]
    assert ffn_bits in [1, 8, 16, 32]

    if ffn_bits == 1:
        ffn_bits =1 
    else:
        ffn_bits = input_bits
    # print("NUM HEADS: ", num_heads)
    attn_flops = dict(
            # Projection 
            kq_proj = 2*2*emb_dim*kq_dim*num_attns,
            kq_proj_bias = 2*kq_dim*num_attns,
            v_proj = 2*emb_dim*v_dim*num_attns,
            v_proj_bias = 2*v_dim*num_attns,

            # Attention
            attn_scores = 2* kq_dim * seq_len * num_attns,
            attn_softmax = SOFTMAX_FLOPS * seq_len * num_heads * num_attns,
            attn_dropout = DROPOUT_FLOPS * seq_len * num_heads * num_attns,
            attn_scale = seq_len * num_heads * num_attns,
            attn_weighted_avg_values = 2 * emb_dim * seq_len * num_attns,

            # Attn_ouput
            attn_output = 2 * emb_dim * emb_dim * num_attns,
            attn_output_bias = emb_dim * num_attns,
            attn_output_dropout = DROPOUT_FLOPS * emb_dim * num_attns,
            attn_output_residual = emb_dim * num_attns,
            attn_output_layer_norm = LAYER_NORM_FLOPS * num_attns,

            # FFNs w/o matrix multiplication
            fc1_act = ACTIVATION_FLOPS * ffn_dim,
            fc1_bias = ffn_dim,
            fc2_bias = emb_dim,
            fc2_dropout = DROPOUT_FLOPS * emb_dim,
            fc2_residual = emb_dim,
            fc2_layer_norm = LAYER_NORM_FLOPS * emb_dim,
            )

    ffn_flops = dict(
            # Matrix multiplications for FFN
            fc1 = 2 * emb_dim * ffn_dim,
            fc2 = 2 * ffn_dim * emb_dim,            
            )
    attn_flops_count = sum(attn_flops.values())
    ffn_flops_count = sum(ffn_flops.values())
    # print(f"{num_attns} | ATTN: {attn_flops_count} | FFN: {ffn_flops_count} | ATTN/FFN: "
    #        f"{attn_flops_count/ffn_flops_count:.2f}")
    res = int(attn_flops_count * input_bits/32) + int(ffn_flops_count * ffn_bits/32)
    res *= (seq_len * bs)
    return res



def get_encoder_layer_flops(**kwargs):
    return get_decoder_layer_flops(**kwargs)


def get_embedding_flops():
    pass

"""
                bs=None,
                kq_dim=None,
                v_dim=None,
                seq_len=None,
                num_heads=None,
                emb_dim=None,
                ffn_dim=None,
                input_bits=None,
                encoder_bits=None,
                decoder_bits=None,
"""

####################### Computing FLOPs END  ###########################

# kqv  o proj: attn_bits
# ffn 1, 2: ffn_bits




def get_params(
                      kq_dim=None,
                      v_dim=None,
                      num_heads=None,
                      emb_dim=None,
                      ffn_dim=None,
                      in_voca_size=None,
                      out_voca_size=None,
                      
                      emb_bits=None,
                      en_attn_bits=None,
                      en_ffn_bits=None,
                      de_attn_bits=None,
                      de_ffn_bits=None,
                      **kwargs,
                      ):
    emb_params = get_emb_params(emb_dim=emb_dim, voca_size=in_voca_size,
                                emb_bits=emb_bits)
    en_params = get_module_params(kq_dim, v_dim,
                                  num_heads, emb_dim, ffn_dim,
                                  en_attn_bits, en_ffn_bits, 1)
    de_params = get_module_params(kq_dim, v_dim,
                                  num_heads, emb_dim, ffn_dim,
                                  de_attn_bits, de_ffn_bits, 2)
    output_params = out_voca_size * emb_dim
    return emb_params + en_params + de_params + output_params




def get_module_params(
                      kq_dim=None,
                      v_dim=None,
                      num_heads=None,
                      emb_dim=None,
                      ffn_dim=None,

                      attn_bits=None,
                      ffn_bits=None,
                      num_attns=None,
                      ):
    res = 0
    for a_bits, f_bits in zip(attn_bits, ffn_bits):
        res += get_layer_params(kq_dim, v_dim,
                                num_heads, emb_dim, ffn_dim,
                                a_bits, f_bits, num_attns)
    return res



def get_layer_params(
                      kq_dim=None,
                      v_dim=None,
                      num_heads=None,
                      emb_dim=None,
                      ffn_dim=None,

                      attn_bits=None,
                      ffn_bits=None,
                      num_attns=None,
                      ):
    assert attn_bits in [1, 8, 16, 32]
    assert ffn_bits in [1, 8, 16, 32]


    attn_params = dict(
            # Projection 
            kq_proj = (kq_dim + 1) * emb_dim * 2,
            v_proj = (v_dim + 1) * emb_dim,

            # Attn_ouput
            attn_output = (emb_dim + 1) * emb_dim,
            )

    ffn_params = dict(
            # Matrix multiplications for FFN
            fc1 = (emb_dim + 1) * ffn_dim,
            fc2 = (ffn_dim + 1) * emb_dim,            
            )

    attn_params_count = int(sum(attn_params.values()) * attn_bits/32) * num_attns
    ffn_params_count = int(sum(ffn_params.values()) * ffn_bits/32)

    res = attn_params_count + ffn_params_count
    return res

def get_emb_params(emb_dim, voca_size, emb_bits):
    assert emb_bits in [1, 8, 16, 32]
    emb_params = dict(
            emb=emb_dim * voca_size
    )
    emb_params_count = int(sum(emb_params.values()) * emb_bits/32)
    return emb_params_count



tf_dict = dict(
                        bs=1,
                        kq_dim= 512,
                        v_dim= 512,
                        seq_len= 100,
                        num_heads= 4,
                        emb_dim= 512,
                        ffn_dim= 1024,

                        in_voca_size=8848,
                        out_voca_size=6632,

                        input_bits = 32,
                        encoder_bits = [32]*6,
                        decoder_bits = [32]*6,

                        emb_bits=32,
                        en_attn_bits=[32]*6,
                        de_attn_bits=[32]*6,
                        en_ffn_bits=[32]*6,
                        de_ffn_bits=[32]*6,

                       ) 
sm_dict = dict(
                        bs=1,
                        kq_dim= 512,
                        v_dim= 512,
                        seq_len= 100,
                        num_heads= 4,
                        emb_dim=512,
                        ffn_dim= 1024,

                        in_voca_size=8848,
                        out_voca_size=6632,

                        input_bits = 16,
                        encoder_bits = [8, 8, 16],
                        decoder_bits = [16, 16, 16],

                        emb_bits=8,
                        en_attn_bits=[8, 8, 32],
                        de_attn_bits=[32, 32, 32],
                        en_ffn_bits=[8, 8, 32],
                        de_ffn_bits=[32, 32],
                       )
              
sm_dict_orig = dict(
                        bs=1,
                        kq_dim= 300,
                        v_dim= 300,
                        seq_len= 100,
                        num_heads= 4,
                        emb_dim= 300,
                        ffn_dim= 512,

                        in_voca_size=8848,
                        out_voca_size=6632,

                        input_bits = 16,
                        encoder_bits = [8, 1, 1, 16],
                        decoder_bits = [16, 16, 16, 16],

                        emb_bits=8,
                        en_attn_bits=[8, 8, 8, 32],
                        de_attn_bits=[32, 32, 32, 32],
                        en_ffn_bits=[1, 1, 8, 32],
                        de_ffn_bits=[32, 32, 32, 32],
                       )

sm_dict_tf = dict(
                        bs=1,
                        kq_dim= 300,
                        v_dim= 300,
                        seq_len= 100,
                        num_heads= 4,
                        emb_dim= 300,
                        ffn_dim= 512,

                        in_voca_size=8848,
                        out_voca_size=6632,

                        input_bits = 32,
                        encoder_bits = [32]*4,
                        decoder_bits = [32]*4,

                        emb_bits=32,
                        en_attn_bits=[32]*4,
                        de_attn_bits=[32]*4,
                        en_ffn_bits=[32]*4,
                        de_ffn_bits=[32]*4,

                       ) 
"""
sm_dict = dict(
                        bs=1,
                        kq_dim= 416,
                        v_dim= 416,
                        seq_len= 100,
                        num_heads= 4,
                        emb_dim= 416,
                        ffn_dim= 640,

                        input_bits = 16,
                        encoder_bits = [1, 1, 8, 16],
                        decoder_bits = [16, 16, 16, 16],
                       )
"""



if __name__ == "__main__":
    tf_flops = get_flops(**tf_dict)
    sm_flops = get_flops(**sm_dict)
    tf_params = get_params(**tf_dict)
    sm_params = get_params(**sm_dict)
    

    print("* FLOPS: ")
    print(f"- tf: {tf_flops} | sm: {sm_flops} | "
          f"tf/sm: {tf_flops/sm_flops:.2f}")
    print("* PARAMS: ")
    print(f"- tf: {tf_params} | sm: {sm_params} | "
          f"tf/sm: {tf_params/sm_params:.2f}")
    

