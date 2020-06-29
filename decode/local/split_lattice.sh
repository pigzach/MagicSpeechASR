#!/bin/bash
cmd='run.pl'
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 3 ];then
  echo 'Usage: $0 <srd-lat-dir> <num-split> <dst-lat-dir>'
  exit 1;
fi

src=$1
num_splits=$2
dest=$3

mkdir -p $dest
echo $num_splits > $dest/num_jobs
gunzip -c $src/lat.*.gz | lattice-copy ark:- ark,scp:$dest/lat.ark,$dest/lat.scp || exit 1;

splits=
for i in `seq 1 $num_splits`;do
  splits="$splits $dest/lat.$i.scp"
done

utils/split_scp.pl $dest/lat.scp $splits
$cmd JOB=1:$num_splits $dest/log/lattice-copy.JOB.log \
  lattice-copy scp:$dest/lat.JOB.scp "ark:|gzip -c > $dest/lat.JOB.gz" || exit 1;
rm -rf $dest/lat.ark $dest/lat.scp $dest/lat.*.scp
echo "$0: Finish split lattice to $num_splits parts."
  
