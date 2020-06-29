#!/usr/bin/env bash
. ./path.sh || exit 1;

tmp=
dir=

if [ $# != 4 ]; then
  echo "Usage: $0 <corpus-data-dir> <dict-dir> <tmp-dir> <output-dir>"
  echo " $0 /export/AISHELL-2/iOS/train data/local/dict data/local/train data/train"
  exit 1;
fi

corpus=$1
dict_dir=$2
tmp=$3
dir=$4

echo "prepare_data.sh: Preparing data in $corpus"

mkdir -p $tmp
mkdir -p $dir

# corpus check
if [ ! -d $corpus ] || [ ! -f $corpus/wav.scp ] || \
   [ ! -f $corpus/utt2spk ] || [ ! -f $corpus/spk2utt ] || \
   [ ! -f $corpus/segments ]; then
  echo "Error: $0 requires wav.scp utt2spk and spk2utt under $corpus directory."
  exit 1;
fi

cp $corpus/{wav.scp,utt2spk,spk2utt,segments} $tmp

# text
if [ -e $corpus/text ];then
  export LC_ALL=en_US.utf-8
  python3 -c "import jieba" 2>/dev/null || \
    (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)
  # jieba's vocab format requires word count(frequency), set to 99
  awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $tmp/word_seg_vocab.txt
  python3 chinese_text_normalization/TN/cn_tn.py --to_upper --has_key $corpus/text $tmp/trans.txt
  python3 local/word_segmentation.py $tmp/word_seg_vocab.txt $tmp/trans.txt > $tmp/text
fi

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text spk2utt utt2spk segments; do
  if [ -e $tmp/$f ];then
    cp $tmp/$f $dir/$f || exit 1;
  fi
done
rm -rf $tmp
utils/fix_data_dir.sh $dir

echo "local/prepare_data.sh succeeded"
exit 0;