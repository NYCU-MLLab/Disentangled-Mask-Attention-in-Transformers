#! /bin/bash
cd examples/translation/
bash prepare-wmt14en2de.sh --icml17
cd ../..
TEXT=examples/translation/wmt14_en_de

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --nwordssrc -1 --nwordstgt -1 \
    --joined-dictionary --workers 20
