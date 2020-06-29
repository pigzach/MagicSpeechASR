#!/bin/bash
nj=32
if [ $# -ne 2 ];then
  echo "Usage: $0 <train-data> <copy-data>"
  exit 1
fi
train=$1
aug=$2

echo "Copy training set to balance data..."
tmp=data/local/copyx3
mkdir -p $tmp
utils/data/copy_data_dir.sh --utt-suffix "-android" $train $tmp/train_copyasandroid
utils/data/copy_data_dir.sh --utt-suffix "-ios" $train $tmp/train_copyasios
utils/data/copy_data_dir.sh --utt-suffix "-recorder" $train $tmp/train_copyasrecorder
utils/combine_data.sh $aug $tmp/train_copyasandroid $tmp/train_copyasios $tmp/train_copyasrecorder
rm -rf $tmp
echo "Done"

exit 0;


