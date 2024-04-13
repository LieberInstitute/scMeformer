#
This repository contains source code of scMeformer for single cell DNAm data imputation.

## Install
Install PyTorch following instructions from https://pytorch.org/ and apex by following https://github.com/NVIDIA/apex.

## Usage

1. Clustering

1.1. Calculate DNAm levels for each 100kb bin and cluster cells based on DNAm levels of 100kb bins.

#Example
```bash
Clusters all cells for a brain sample (Br1092)

$python clustering.py Br1092

```

1.2. Calculate cluster features for each CpG site based on cell clusters produced by step 1.1

#Example
```bash
Construct training data and calculate cluster features for a brain sample (Br1092)

$python run_feature.py Br1092

```


2. Training

2.1. Train DNAm prediction model using single cell data. We trained a prediction model for each brain sample. 

#Example
```bash
train the DNAm prediction model for one brain sample (Br1092) using four GPUs

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer single_cell_regression \
	--exp_name single_cell_regression \
	--learning_rate 0.000176 \
	--batch_size 128 \
	--data_dir ./datasets/Schizo_Control/Br1092 \
	--output_dir ./outputs/Schizo_Control/Br1092 \
	--warmup_steps 10000 \
	--gradient_accumulation_steps 1 \
	--fp16 --local_rank 0 \
	--nproc_per_node 4 \
	--model_config_file ./config/config.json
```
"num_features" and "task_size" in "./config/schizo_control/Br1092/config.json" denote the numbers of clusters and cells in the brain sample (Br1092).

3. Prediction

3.1. Predicts DNAm levels of CpGs from DNA sequence using one GPU.
#Example
```bash
predict DNAm levels of CpGs for all cells in a brain sample (Br1092) using the trained model

CUDA_VISIBLE_DEVICES=0 python3 main.py transformer single_cell_prediction \
	--exp_name single_cell_prediction \
	--batch_size 1024 \
	--learning_rate 0.000176 \
	--fp16 \
	--warmup_steps 10000 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/genome_cpg \
	--output_dir ./outputs/prediction/schizo_control/Br1092/chr1 \
	--num_train_epochs 500 \
	--model_config_file ./config/config.json \
	--from_pretrained ./outputs/Schizo_Control/Br1092 \
	--split chr1
```

4. Demo

4.1 Training data from snmCAT-seq data include following four files:

4.1.1 ./datasets/methylation_data/chr1.json contains 10000 CpGs and each CpG have five keys:
(1) chromosome, (2) position, (3) strand, (4) cells with hypermethylation, and (5) cells with hypomethylation

4.1.2 ./scMeformer/datasets/feature_data/chr1.npy contains DNAm levels of 93 clusters for above 10000 CpGs

4.1.3 ./scMeformer/datasets/genome/chr1.npy contains one-hot encoded genome sequences covering above 10000 CpGs

4.1.4 ./scMeformer/datasets/position/chr1.npy provides the index of each CpG in ./scMeformer/datasets/feature_data/chr1.npy by its position.


4.2 Validation and test data include CpGs on chromosomes 21 and 22, respectively. Both include four files of the same format with training data.

4.2.1 Both validation and test data consist of 2000 CpGs.

4.3 Train DNAm imputation model by this demo data.

```bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer single_cell_regression \
        --exp_name single_cell_regression \
        --learning_rate 0.000176 \
        --batch_size 128 \
        --data_dir ./datasets/ \
        --output_dir ./scMeformer/outputs/demo_model \
        --warmup_steps 10000 \
        --gradient_accumulation_steps 1 \
        --fp16 --local_rank 0 \
        --nproc_per_node 4 \
        --model_config_file ./config/config.json
```
