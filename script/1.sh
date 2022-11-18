CUDA_VISIBLE_DEVICES=1 python ../src/pruning.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch spt_iwslt_de_en --share-decoder-input-output-embed \
    --task SPTtranslation \
    --optimizer spt_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion spt --label-smoothing 0.1 \
    --max-epoch 1000 \
    --compression-rate 0.1 --srp --pruning-stage 1 \
    --save-interval 100 \
    --pruning-iter 1 --pruning-period 1000 --decreasing sa \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/10_1000_ts \
    --pretrained-model ../checkpoints/1/checkpoint_best.pt \
    --restore-file ../checkpoints/10_1000_ts/checkpoint700.pt \

CUDA_VISIBLE_DEVICES=1 python ../src/pruning.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch spt_iwslt_de_en --share-decoder-input-output-embed \
    --task SPTtranslation \
    --optimizer spt_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion spt --label-smoothing 0.1 \
    --compression-rate 0.1 --srp --pruning-stage 2 \
    --save-interval 100 \
    --pruning-iter 1 --pruning-period 500 --decreasing sa \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/10_1000_ts_2 \
    --pretrained-model ../checkpoints/10_1000_ts/checkpoint_last.pt
