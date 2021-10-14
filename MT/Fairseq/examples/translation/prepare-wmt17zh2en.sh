#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Reference: https://blog.csdn.net/Treasure003/article/details/107285835

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32768

URLS=(
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://data.statmt.org/wmt17/translation-task/test.tgz"
)
FILES=(
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test.tgz"
)
CORPORA=(
    "training/news-commentary-v12.zh-en"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=zh
tgt=en
lang=zh-en
prep=wmt17_zh_en
tmp=$prep/tmp
orig=wmt17_orig
dev=dev/newstest2013

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd ..

# Preprocess src (Chinese)
echo "pre-processing train data..."
rm $tmp/train.tags.$lang.tok.$src
for f in "${CORPORA[@]}"; do
    cat $orig/$f.$src | \
        perl $NORM_PUNC $src | \
        perl $REM_NON_PRINT_CHAR | \
        python3 -m jieba -d " " >> $tmp/train.tags.$lang.tok.$src
done

# # Preprocess tgt (English)
rm $tmp/train.tags.$lang.tok.$tgt
for f in "${CORPORA[@]}"; do
    cat $orig/$f.$tgt | \
        perl $NORM_PUNC $tgt | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $tgt >> $tmp/train.tags.$lang.tok.$tgt
done

echo "pre-processing test data..."
rm $tmp/test.$src
rm $tmp/test.$tgt

for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi

    if [ "$l" == "zh" ]; then
        grep '<seg id' $orig/test/newstest2017-zhen-$t.$l.sgm | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
            python3 -m jieba -d " " >> $tmp/test.$l
    else
        grep '<seg id' $orig/test/newstest2017-zhen-$t.$l.sgm | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
            perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    fi
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.zh-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python3 $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python3 $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done
