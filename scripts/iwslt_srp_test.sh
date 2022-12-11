CUDA_VISIBLE_DEVICES=0 python ../src/generate.py ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src --arch srp_iwslt_de_en --task SRPtranslation \
    --weighted-layernorm \
    --path ../checkpoints/base/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
