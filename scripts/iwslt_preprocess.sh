# Download and prepare the data
bash prepare-iwslt14.sh

# Preprocess/binarize the data
TEXT=../data-bin/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir ../data-bin/iwslt14.tokenized.de-en \
    --workers 20
