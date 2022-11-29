# SRP (Selectively Regularized Pruning)
This project is a PyTorch implementation of SRP (Selectively Regularized Pruning) on Transformer. SRP proposes a novel approach that improves structured pruning performance. This package is especially for Transformer model.  

## Overview
#### Brief Explanation of SRP. 
SRP proposed a novel process for pruning Transfomer and the process works as following three steps.

0) Defining Pruning-safe Architecture
We first define an architecture is pruning-safe for some parameters under some codition iff the inference of the model is consistent after pruning the parameters under the condition.

1) Designing Pruning-safe Architecture
We modify the architecture of Transformer to be pruning-safe. We introduce connectivity parameters and weighted layer normalization. 
Our modified architecture is pruning safe when pruning parameters that its corresponding connectivity parameters are zero.

2) Selecting Parameters to be pruned
We compute the negative partial derivative of the objective function with respect to each connectivity parameter and use as for the importance score of the corresponding parameter. 
We select parameters with the lowest importance score to be pruned.

3) Shrinking Paramters with Selective Regularization
We freeze the selected parameters and continuously shrinking corresponding connectivity parameters. 
SRP proposes two types of shrinking strategies: arithmetic and geometrical shrinking.
SRP proposed two-staged pruning which is a novel pruning strategy that prunes parameters within two stages.

#### Code Description
This repository is based on the [FAIRSEQ](https://github.com/facebookresearch/fairseq).
All source files are from the repository if not mentioned otherwise.
The overall process 

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
  │     │    ├── srp_config.py: configurations of SRP
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
  │     ├── train.py : training a new model 
  │     ├── trainer.py : codes for managing training process 
  │     ├── pruning.py : pruning the pre-trained model
  │     ├── finetuning.py : finetuning the pruned model
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
  │     ├── iwslt_srp_two_staged_pruning.sh a script for performing two-staged pruning 
  │     └── iwslt_srp_test.sh: a script for testing the trained model 
  │  
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
Install [PyTorch](http://pytorch.org/) (version >= 1.5.0) and install fairseq by following instructions:
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
Followings are key arguments to 

* First, we begin with training a Transformer model
```
cd scripts
bash iwslt_srp.sh
```
This code is alaso saved as scripts/iwslt_srp.sh

* To perform pruning using SRP, run script:
```
cd scripts
bash iwslt_srp_pruning.sh
```
This code is alaso saved as scripts/iwslt_srp_two_staged_pruning.sh

* If you want to perform single-staged pruning, run script:
```
cd scripts
bash iwslt_srp_pruning.sh
```
This code is alaso saved as scripts/iwslt_srp_two_staged_pruning.sh

* To perform finetuning after pruning, run script:
```
cd scripts
bash iwslt_srp_pruning.sh
```
This code is alaso saved as scripts/iwslt_srp_two_staged_pruning.sh

* To testing after pruning, run script:
```
cd scripts
bash iwslt_srp_pruning.sh
```
This code is alaso saved as scripts/iwslt_srp_two_staged_pruning.sh


## Reference
* FAIRSEQ: https://github.com/facebookresearch/fairseq



## Run your own training  
* We provide an example how to run the codes. We use task: 'MRPC', teacher layer: 12, and student layer: 3 as an example.
* Before starting, we need to specify a few things.
    * task: one of the GLUE datasets
    * train_type: one of the followings - ft, kd, pkd 
    * model_type: one of the followings - Original, SPS
    * student_hidden_layers: the number of student layers
    * train_seed: the train seed to use. If default -> random 
    * saving_criterion_acc: if the model's val accuracy is above this value, we save the model.
    * saving_criterion_loss: if the model's val loss is below this value, we save the model.
    * load_model_dir: specify a directory of the checkpoint if you want to load one.
    * output_dir: specify a directory where the outputs will be written and saved.
    
* First, We begin with finetuning the teacher model
    ```
    run script
    python src/finetune.py \
    --task 'MRPC' \
    --train_type 'ft' \
    --model_type 'Original' \
    --student_hidden_layers 12 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0 .6 \
    --output_dir 'run-1'
    ```
    The trained model will be saved in 'src/data/outputs/KD/{task}/teacher_12layer/'

* To use the teacher model's predictions for PTP, KD, and PKD run script:
    ```
    python src/save_teacher_outputs.py
    ```
    The teacher predictions will be saved in 'src/data/outputs/KD/{task}/{task}_normal_kd_teacher_12layer_result_summary.pkl'
    or 'src/data/outputs/KD/{task}/{task}_patient_kd_teacher_12layer_result_summary.pkl'

* To apply PTP to the student model, run script:
    ```
    run script:
    python src/PTP.py \
    --task 'MRPC' \
    --train_type 'ft' \
    --model_type 'SPS' \
    --student_hidden_layer 3 \
    --saving_criterion_acc 0.8 \
    --output_dir 'run-1'
    ```
    The pretrained student model will be saved in 'src/data/outputs/KD/{task}/teacher_12layer/'. 
    you may specify the hyperparameter 't' in src/utils/nli_data_processing.py line 713~.
* When PTP is done, we can finally finetune the student model by running script:
    ```
    python src/finetune.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'SPS' \
    --student_hidden_layers 3 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.6 \
    --load_model_dir 'run-1/PTP.encoder_loss.pkl' \
    --output_dir 'run-1/final_results'
    ```
