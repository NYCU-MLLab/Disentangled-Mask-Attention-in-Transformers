# Disentangled Mask Attention in Transformers
This repository is developed under SpeechBrain Framework

## Experimental results
The configurations of Transformer and Disentangled Transformer are listing as following.
|Model|Layers|Heads|Embed|FFN|Clusters|Params|
|:----|:----:|:---:|:---:|:-:|:------:|:----:|
|Transformer|6/6|4|256|1024|-|16.8M|
|Disentangled Transformer|6/6|4|256|1024|8/4|16.9M|

The experiment results on Aishell-1 benchmark are listed as following.

| Dataset       | Task           | System  | Dev CER | Test CER | Learning Ratio | Pretrained weights |
| ------------- |:-------------:| -----:|-----:|:-:|:-:|:-:|
| Aishell-1     | Speech Recognition | CNN + Transformer | 7.70% | 8.15% | 1.0x | [Download](https://drive.google.com/file/d/1Rl8At8_DaWvtdjpOgV1WasUJlQ3tZ56g/view?usp=sharing) |
| Aishell-1     | Speech Recognition | CNN + Disentangled Transformer | 7.51% | 7.89% | 2.0x | [Download](https://drive.google.com/file/d/1qVUaUnEFLE_WSBabC2HmP6t_1CC7zoM_/view?usp=sharing) |

## Installatation
1. Install SpeechBrain
```
> cd speechbrain
> pip3 install -r requirements.txt
> pip3 install --editable .
```

## Model training (Aishell-1)
```
> cd recipes/AISHELL-1/ASR/transformer/
```

1. Train Transformer
```
> python3 train.py hparams/train_ASR_transformer.yaml --data_folder=./data
```

2. Train Disentangled Transformer
```
> python3 train_disentangled_transformer.py hparams/train_ASR_disentangled_transformer.yaml --data_folder=./data
```

The results will be saved in the `output_folder` specified in the yaml file. Both detailed logs and experiment outputs are saved there.

## Performance evaluation (Aishell-1, Test CER)
```
> cd recipes/AISHELL-1/ASR/transformer/
```
* Evaluating Transfomer
```
> python3 train.py hparams/train_ASR_transformer.yaml --data_folder=./data --output_folder=results/transformer/8886 --eval_only
```
* Evaluating Disentanlged Transfomer
```
> python3 train_disentangled_transformer.py hparams/train_ASR_disentangled_transformer.yaml --data_folder=./data --output_folder=results/<model_folder>/8886 --eval_only
``` 
* The result is saved in `recipes/AISHELL-1/ASR/transformer/results/<model_folder>/cer.txt`

### Usage of pretrained weights
* Copy pretrained weights folder to `recipes/AISHELL-1/ASR/transformer/results/`
* Replace `<model_folder>` with the root folder name of pretrained weight, ex. `disentangled_transformer_lr-2`.

