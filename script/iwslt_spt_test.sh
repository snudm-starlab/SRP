CUDA_VISIBLE_DEVICES=1 python ../src/generate.py ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src --arch spt_iwslt_de_en --task SPTtranslation \
    --weighted-layernorm \
    --path ../checkpoints/10_1000_ts_wl_2/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
