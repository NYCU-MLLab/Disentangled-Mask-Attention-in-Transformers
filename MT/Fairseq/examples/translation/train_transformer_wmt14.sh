#!/usr/bin/env bash
dataset=${dataset:-wmt14_en_de}
dropout=${dropout:-0.1}
weight_deacy=${weight_deacy:-0.0000}
lr=${lr:-7e-4}
max_tokens=${max_tokens:-4096}
max_tokens_valid=${max_tokens_valid:-4096}
update_freq=${update_freq:-8}
arch=${arch:-disentangled_transformer_wmt_en_de}
mi_loss_weight=${mi_loss_weight:-1.}
max_epochs=${max_epochs:-30}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi

    shift
done

data_dir=data-bin/${dataset}
mkdir -p checkpoints/${dataset}/transformer

PYTHONIOENCODING=utf-8 fairseq-train \
    ${data_dir} \
    --user-dir examples/translation/disentangled_transformer \
    --arch ${arch} --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout ${dropout} --weight-decay ${weight_deacy} \
    --mi_loss_weight ${mi_loss_weight} \
    --criterion disentangled_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens ${max_tokens} \
    --max-tokens-valid ${max_tokens_valid} \
    --update-freq ${update_freq} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/${dataset}/transformer/ckpt \
    --tensorboard-logdir checkpoints/${dataset}/transformer/tensorboard \
    --log-format json \
    --keep-last-epochs 5 \
    --max-epoch ${max_epochs} \
    --fp16 \
    --unk-augmentation-rate 0.0 \
    --regular-attn \
    | tee checkpoints/${dataset}/transformer/train.log
