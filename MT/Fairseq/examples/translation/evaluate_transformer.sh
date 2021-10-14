#! /bin/bash
dataset=${dataset:-iwslt14.tokenized.de-en}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi

    shift
done

mkdir -p checkpoints/${dataset}/transformer/evaluate

python3 scripts/average_checkpoints.py \
    --inputs checkpoints/${dataset}/transformer/ckpt \
    --output checkpoints/${dataset}/transformer/ckpt/checkpoint_avg_last_5.pt \
    --num-epoch-checkpoints 5

PYTHONIOENCODING=utf-8 fairseq-generate data-bin/${dataset} \
    --path checkpoints/${dataset}/transformer/ckpt/checkpoint_avg_last_5.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --user-dir examples/translation/disentangled_transformer \
    | tee checkpoints/${dataset}/transformer/evaluate/evaluate.log
