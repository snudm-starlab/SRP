# SRP (Selectively Regularized Pruning)
This project is a PyTorch implementation of SRP (Selectively Regularized Pruning) on Transformer. SRP proposes a novel approach that improves structured pruning performance. This package is especially for Transformer model.  

## Overview
#### Brief Explanation of SRP. 
SRP proposed a novel process for pruning Transfomer and works as following three steps.

#### 0. Defining Pruning-safe Architecture
Before we start pruning process, we define that an architecture is pruning-safe for some parameters under some codition iff the inference of the model is consistent after pruning the parameters under the condition.

#### 1. Designing Pruning-safe Architecture

We modify the architecture of Transformer to be pruning-safe. We introduce connectivity parameters and weighted layer normalization. 
Our modified architecture is pruning safe when pruning parameters that its corresponding connectivity parameters are zero.

#### 2. Selecting Parameters to be pruned

We compute the negative partial derivative of the objective function with respect to each connectivity parameter and use it as the importance score of the corresponding parameter. 
We select parameters with the lowest importance score to be pruned.

#### 3. Shrinking Paramters with Selective Regularization

We freeze the selected parameters and continuously shrinking corresponding connectivity parameters. 
SRP proposes two types of shrinking strategies: arithmetic and geometrical shrinking.
SRP proposed two-staged pruning which is a novel pruning strategy that prunes parameters within two stages.

#### Code Description
This repository is written based on the codes in [FAIRSEQ](https://github.com/facebookresearch/fairseq).
Here's an overview of our codes.

``` Unicode
SRP
  │
  ├──  src    
  │     ├── criterions
  │     │    └── srp.py: customized criterion for SRP
  │     ├── models
  │     │    ├── srp_base.py: SRP model 
  │     │    ├── srp_encoder.py: SRP encoder
  │     │    ├── srp_decoder.py: SRP decoder
  │     │    ├── srp_config.py: configurations for SRP
  │     │    └── srp_legacy.py: the legacy implementation of the srp models
  │     ├── modules
  │     │    ├── layer_norm.py: codes for weighted layer normalization
  │     │    ├── multihead_attention.py: modified codes for multihead attention
  │     │    └── srp_layer.py : layers of encoder and decoder of SRP
  │     ├── optim
  │     │    └── SRPadam.py : customized Adam optimizer for SRP
  │     ├── tasks
  │     │    └── SRPtranslation : customized translation task for SRP
  │     │    
  │     ├── train.py : codes for training a new model 
  │     ├── trainer.py : codes for managing training process 
  │     ├── pruning.py : codes for pruning the pre-trained model
  │     ├── finetuning.py : codes for finetuning the pruned model
  │     ├── generate.py : Translate pre-processed data with a trained model
  │     ├── sequence_generator.py : Codes for generating sequences
  │     ├── checkpoint_utils.py : Codes for saving and loading checkpoints
  │     └── flops_counter.py: Codes for computing FLOPs of the model
  │     
  │     
  ├──  scripts
  │     ├── iwslt_preprocess.sh: a script for downloading and preprocess iwslt14
  │     ├── prepare-iwslt14.sh : a script for preparing iwslt14 which is run by iwslt_preprocess.sh
  │     ├── iwslt_srp.sh : a script for training a baseline model
  │     ├── iwslt_srp_pruning.sh : a script for pruning the pre-trained model
  │     ├── iwslt_srp_two_staged_pruning.sh : a script for performing two-staged pruning 
  │     ├── iwslt_srp_finetuning.sh : a script for finetuning pruned model 
  │     ├── iwslt_srp_test.sh: a script for testing the trained model
  │     └── demo.sh : a script for running demo  
  │     
  ├──  data-bin : a directory for saving datasets
  ├──  checkpoints : a directory for saving checkpoints 
  │  
  ├── Makefile
  ├── LICENSE
  └── README.md

```

## Install 

#### Environment 
* Ubuntu
* CUDA 11.6
* numpy 1.23.4
* torch 1.12.1
* sacrebleu 2.0.0
* pandas 

## Installing Requirements
Install [PyTorch](http://pytorch.org/) (version >= 1.5.0) and install fairseq with following instructions:
```
git clone https://github.com/pytorch/fairseq 
cd fairseq
git reset --hard 0b54d9fb2e42c2f40db3449ca34586952b8abe94
pip install --editable ./
pip install sacremoses
```

# Getting Started

## Preprocess
Download IWSLT'14 German to English dataset by running script:
```
cd scripts
bash iwslt_preprocess.sh
```

## Demo 
you can run the demo version.
```
make
```

## Run Your Own Training
* We provide scripts for pre-training, pruning and testing.
Followings are key arguments:
    * arch: architecture type
    * compression-rate: target compreesion rate
    * srp: whether use srp or not
    * pruning-stage: the stage of pruning, use 0 for single-staged pruning
    * pruning-iter: number of pruning iterations
    * pruning-period: number of epochs for pruning process
    * decreasing: decreasing type for connectivity parameters
    * save-dir: path for saving checkpoints
    * pretrained-model: path for pre-trained model to be pruned
    

* First, we begin with training a Transformer model
```
CUDA_VISIBLE_DEVICES=0 python ../src/train.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch srp_iwslt_de_en --share-decoder-input-output-embed \
    --task SRPtranslation \
    --optimizer srp_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion srp --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/base \
```
This code is also saved in scripts/iwslt_srp.sh

* To perform pruning using SRP, run following scripts.

For stage 1: 
```
CUDA_VISIBLE_DEVICES=0 python ../src/pruning.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch srp_iwslt_de_en --share-decoder-input-output-embed \
    --task SRPtranslation \
    --optimizer srp_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion srp --label-smoothing 0.1 \
    --max-epoch 20 --weighted-layernorm \
    --compression-rate 0.2 --srp --pruning-stage 1 \
    --save-interval 1 \
    --pruning-iter 1 --pruning-period 20 --decreasing sa \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/stage1 \
    --pretrained-model ../checkpoints/base/checkpoint_best.pt \

```

For stage 2: 


```
CUDA_VISIBLE_DEVICES=0 python ..src/pruning.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch srp_iwslt_de_en --share-decoder-input-output-embed \
    --task SRPtranslation \
    --optimizer srp_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion srp --label-smoothing 0.1 \
    --compression-rate 0.2 --srp --pruning-stage 2 \
    --save-interval 1 --weighted-layernorm \
    --pruning-iter 1 --pruning-period 20 --decreasing sa \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/stage2 \
    --pretrained-model ../checkpoints/stage1/checkpoint_last.pt
```

These codes are also saved in scripts/iwslt_srp_two_staged_pruning.sh

* If you want to perform single-staged pruning, run script:
```
CUDA_VISIBLE_DEVICES=0 python ../src/pruning.py \
    ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src \
    --arch srp_iwslt_de_en --share-decoder-input-output-embed \
    --task SRPtranslation \
    --optimizer srp_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion srp --label-smoothing 0.1 \
    --compression-rate 0.2 --srp --pruning-stage 0 \
    --pruning-iter 1 --pruning-period 20 --decreasing sa \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/single_stage \
    --pretrained-model ../checkpoints/base/checkpoint_best.pt

```
This code is also saved in scripts/iwslt_srp_two_staged_pruning.sh

* To perform finetuning after pruning, run script:
```
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
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ../checkpoints/finetuned \
    --pretrained-model ../checkpoints/stage2/checkpoint_last.pt
```
This code is also saved in scripts/iwslt_srp_two_staged_pruning.sh

* To testing after pruning, run script:
```
CUDA_VISIBLE_DEVICES=0 python ../src/generate.py ../data-bin/iwslt14.tokenized.de-en \
    --user-dir ../src --arch srp_iwslt_de_en --task SRPtranslation \
    --weighted-layernorm \
    --path ../checkpoints/base/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```
This code is also saved as scripts/iwslt_srp_two_staged_pruning.sh


## Reference
* FAIRSEQ: https://github.com/facebookresearch/fairseq
