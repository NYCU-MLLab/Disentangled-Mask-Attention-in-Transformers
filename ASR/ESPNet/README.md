# Disentangled Mask Attention in Transformers
This repository is developed under ESPnet framework.

## Experimental results
The experiment results on Aishell-1 benchmark are listed as following.

|Model|Dev CER|Test CER|LR|HR|Learning ratio|Pretrained weights|
|:----|:-----:|:------:|:-:|:-:|:------:|:--------------------:|
|Transformer|7.6|8.4|0.55|0.43|1x|[Download](https://drive.google.com/file/d/1twT47kgKYwUiAO1HJFdN0kFvZoItJMvg/view?usp=sharing)|
|Disentangled Transformer|7.2|8.1|0.52|0.40|2x|[Download](https://drive.google.com/file/d/1U2Lkr23X9_yomdVUIyL7-MAwS3bq7NZY/view?usp=sharing)|

> LR: Layer Redundancy  
> HR: Head Redundancy

## Installation
1. Install required packages
```
> apt-get install gfortran libtool gawk libsndfile1-dev ffmpeg flac libatlas-base-dev \
                bc build-essential libboost-all-dev libbz2-dev liblzma-dev subversion
```

2. Build ESPnet
```
> cd tools/
> make TH_VERSION=1.9.0
```
> Tested pytorch version: 1.7.1+cu110, 1.9.0+cu111

3. Build kaldi
```
> cd kaldi/tools/
> make -j 8

> ./extras/install_mkl.sh

> cd ../src
> ./configure --use-cuda=no
> make -j clean depend; make -j 8
```

Followed the documentation on espnet official site.

## Data preprocessing
1. Modfiy `data` varaible in `egs/aishell/asr1/run.sh` to the location which the Aishell-1 dataset will be downloaded, and make sure the path exists.
2. Run following commands
    ```
    > cd egs/aishell/asr1
    > ./run.sh --stage -1
    ```
    > If `stop_stage` is not specified in argument, model will start training after preprocessing.

## Model training
```
> cd egs/aishell/asr1
> ./run.sh --stage 4
```

## Performance evaluation
```
> cd egs/aishell/asr1
> ./run.sh --stage 5
```

### Usage of pretrained weights
1. Copy all the files in pretrained weights to `egs/aishell/asr1/exp/<model_tag>/results/`, where `<model_tag>` should be replaced with one of followings.
    * `train_sp_pytorch_transformer-6-1024-20`
    * `train_sp_pytorch_disentangled_transformer-6-1024-lr-2`
2. Modify `tag` variable in `egs/aishell/asr1/run.sh` to `<model_tag>`.
3. If skip training stage and evaluate the performance directly, the line `234` to `243` in `egs/aishell/asr1/run.sh` need to be commented.
