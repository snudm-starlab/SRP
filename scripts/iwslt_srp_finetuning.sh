CUDA_VISIBLE_DEVICES=0 python ../src/finetuning.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch srp_iwslt_de_en --share-decoder-input-output-embed \
    --task SRPtranslation \
    --optimizer srp_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion srp --label-smoothing 0.1 \
    --max-epoch 20 \
    --attn-kd 1 --prob-kd 0. --T 10 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/fttest \
    --pretrained-model ../checkpoints/p2/checkpoint_last.pt
