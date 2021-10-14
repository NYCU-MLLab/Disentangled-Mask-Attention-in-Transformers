#! /bin/bash

cd examples/translation/
bash prepare-wmt17zh2en.sh
cd ../..
TEXT=examples/translation/wmt17_zh_en

fairseq-preprocess \
    --source-lang zh --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_zh_en --thresholdtgt 0 --thresholdsrc 0 \
    --nwordssrc 32768 --nwordstgt 32768 \
    --workers 20
