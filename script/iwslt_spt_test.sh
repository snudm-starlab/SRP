python ../src/generate.py ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src --arch spt_iwslt_de_en --task SPTtranslation \
    --path ../checkpoints/spt_base_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
