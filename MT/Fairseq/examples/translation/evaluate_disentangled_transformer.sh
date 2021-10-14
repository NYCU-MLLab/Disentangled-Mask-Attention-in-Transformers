#! /bin/bash
grad_multiplier=${grad_multiplier:-10.}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi

    shift
done

dataset=${dataset:-iwslt14.tokenized.de-en}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi

    shift
done

mkdir -p checkpoints/${dataset}/disentangled_transformer/evaluate

python3 scripts/average_checkpoints.py \
    --inputs checkpoints/${dataset}/disentangled_transformer/ckpt \
    --output checkpoints/${dataset}/disentangled_transformer/ckpt/checkpoint_avg_last_5.pt \
    --num-epoch-checkpoints 5

PYTHONIOENCODING=utf-8 fairseq-generate data-bin/${dataset} \
    --path checkpoints/${dataset}/disentangled_transformer/ckpt/checkpoint_avg_last_5.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --user-dir examples/translation/disentangled_transformer \
    --model-overrides "{'grad_multiplier': ${grad_multiplier}}" \
    | tee checkpoints/${dataset}/disentangled_transformer/evaluate/evaluate.log
