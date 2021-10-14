# Disentangled Mask Attention in Transformers
This repository is developed under Fairseq framework.

## Installation
```
> pip install --editable ./ 
> pip install -r requirements.txt
```
```
> git clone https://github.com/NVIDIA/apex
> cd apex
> pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

> Note: CUDA version=11.1

## Data preprocessing
* IWSLT'14 De-En
```
> ./examples/translation/preprocess_iwslt14_de_en.sh
```
* WMT'14 En-DE
```
> ./examples/translation/preprocess_wmt14_en_de.sh
```
* WMT'17 Zh-En
```
> ./examples/translation/preprocess_wmt17_zh_en.sh
```

## Model training
* IWSLT'14 De-En
```
> ./examples/translation/train_disentangled_transformer.sh
```
* WMT'14 En-De
```
> ./examples/translation/train_disentangled_transformer.sh \
  --dataset wmt14_en_de
```
* WMT'17 Zh-En
```
> ./examples/translation/train_disentangled_transformer.sh \
  --dataset wmt17_zh_en
```

## Performance evaluation
* IWSLT'14 De-En
```
> ./examples/translation/evaluate_disentangled_transformer.sh
```
* WMT'14 En-De
```
> ./examples/translation/evaluate_disentangled_transformer.sh \
  --dataset wmt14_en_de
> bash ./scripts/compound_split_bleu.sh \
  ./checkpoints/wmt14_en_de/disentangled_transformer/evaluate/evaluate.log
```
* WMT'17 Zh-En
```
> ./examples/translation/evaluate_disentangled_transformer.sh \
  --dataset wmt17_zh_en
```

## Experimental results

* IWSLT'14 De-En

  |Model|BLEU|LR|HR|Pretrained weights|
  |:----|:--:|:-:|:-:|:----------------:|
  |Transformer|34.50|0.74|0.65|[Download](https://drive.google.com/file/d/11nJz89ei0xx_CUDKhB80C47klMcaFUs2/view?usp=sharing)|
  |DT (SMA only)|34.66|0.73|0.64|[Download](https://drive.google.com/file/d/1-gd40BiKPnAzG4eU0KiPyHW7YEk8MU2D/view?usp=sharing)|
  |DT|35.06|0.60|0.51|[Download](https://drive.google.com/file/d/1urywQNRbwYMwc0kPosy2PliE4wip9WWk/view?usp=sharing)|
  |DT* (base)|35.31|0.62|0.53|[Download](https://drive.google.com/file/d/1PSYNlpi6ejQNZbacSKpEo8O8M2HnKkJ8/view?usp=sharing)|
  |DT* (tiny)|34.96|0.63|0.58|[Download](https://drive.google.com/file/d/1IxdWbrdyWXU3XXh0PBU_Z8PiQZ03rK4I/view?usp=sharing)|

  > DT: Disentangled Transformer  
  > LR: Layer Redundancy  
  > HR: Head Redundancy  
  > SMA: Semantic Mask Attention

* WMT'14 En-De

  |Model|BLEU|LR|HR|Pretrained weights|
  |:----|:--:|:-:|:-:|:----------------:|
  |Transformer|27.75|0.79|0.69|[Download](https://drive.google.com/file/d/1wWGdwCODrtmkwFdzBpHCQTlSUncOqffB/view?usp=sharing)|
  |DT* (base)|28.35|0.73|0.58|[Download](https://drive.google.com/file/d/1_IjdZYMRkJSJ91JeGqD4vnJeI9EO5ncA/view?usp=sharing)|
  |DT* (small)|28.16|0.70|0.57|[Download](https://drive.google.com/file/d/1EMVMl0FotvQe_HP-pEsmrZ_gKQ4AUkLK/view?usp=sharing)|


* WMT'17 Zh-En

  |Model|BLEU|LR|HR|Pretrained weights|
  |:----|:--:|:-:|:-:|:----------------:|
  |Transformer|12.76|0.71|0.60|[Download](https://drive.google.com/file/d/1943OZsOabyHMn-FftJzRvHYfMUkbzDIo/view?usp=sharing)|
  |DT* (base)|13.13|0.62|0.55|[Download](https://drive.google.com/file/d/1M-NWaeccBszwD-aD7jnRI_t6Og1Rwfmt/view?usp=sharing)|

### Usage of pretrained weight
  1. Download pretrained weights
  2. Copy `checkpoint.pt` or `checkpoint_avg_last_5.pt` to `./checkpoints/<task>/<model>/ckpt/checkpoint_avg_last_5.pt`, where `<task>` need to be changed to one of followings,  
    * `iwslt14.tokenized.de-en`  
    * `wmt14_en_de`  
    * `wmt17_zh_en`  
  and `<model>` need to be one of followings.  
    * `transformer`  
    * `disentangled_transformer`
  3. To evaluate `DT` or `DT (SMA only)` on IWSLT'14, please use following commnad
  ```
  ./examples/translation/evaluate_disentangled_transformer.sh \
  --grad_multiplier 1.
  ```
