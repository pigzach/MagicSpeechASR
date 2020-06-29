#!/bin/bash
stage=0
cmd=run.pl
beam=8.0
wip=1.0
ac_scale=12
weight1=0.55
weight2=0.75
lattice_depth=150
scoring_opts='--decode-mbr true'
#decode_mbr=true

. ./path.sh
. ./utils/parse_options.sh

data=$1
lang_or_graph=$2
latdir=$3
dir=$4

if [ $# -ne 4 ];then
  echo "Usage: $0 <data> <lang> <latdir> <dir>"
  exit 1
fi

mkdir -p $dir
if [ $stage -le 0 ];then
  gunzip -c $latdir/lat.*.gz | lattice-copy ark:- ark,scp:$dir/lat.ark,$dir/lat.scp
  cat $dir/lat.scp | sed 's#_Android\|_IOS\|_Recorder##g' | awk '{print $1}' | awk -F '-' '{print $0" "$1}' | sort -u > $dir/utt2spk
  utils/utt2spk_to_spk2utt.pl <$dir/utt2spk > $dir/spk2utt
  nj=`cat $dir/spk2utt | wc -l`
  echo $nj > $dir/num_jobs
  for device in Android IOS Recorder;do
    cat $dir/lat.scp | grep $device | sed "s#_$device##g" | sort -k1,1 > $dir/$device.lat.scp
  done
  for device in IOS Recorder Android;do
    for i in `seq $nj`;do
      utils/split_scp.pl -j $nj $i --one-based --utt2spk=$dir/utt2spk $dir/$device.lat.scp $dir/$device.lat.$i.scp
    done
  done
fi

if [ $stage -le 1 ];then
  #lattice-combine scp:$dir/IOS.lat.scp scp:$dir/Android.lat.scp "ark:|gzip -c > $dir/lat.1.gz"
  scale=`python -c "print (1.0 / $ac_scale)"`
  $cmd JOB=1:$nj $dir/log/lattice_combine_ios_recorder.JOB.log \
    lattice-interp --alpha=$weight1 "ark:lattice-limit-depth --acoustic-scale=$scale --max-arcs-per-frame=$lattice_depth scp:$dir/IOS.lat.JOB.scp ark:- |" \
      "ark:lattice-limit-depth --acoustic-scale=$scale --max-arcs-per-frame=$lattice_depth scp:$dir/Recorder.lat.JOB.scp ark:- |" ark:- \| \
    lattice-copy-backoff scp:$dir/IOS.lat.JOB.scp ark:- ark,scp:$dir/IOS+Recorder.lat.JOB.ark,$dir/IOS+Recorder.lat.JOB.scp || exit 1;

  $cmd JOB=1:$nj $dir/log/lattice_combine_ios_recorder_android.JOB.log \
    lattice-interp --alpha=$weight2 "ark:lattice-limit-depth --acoustic-scale=$scale --max-arcs-per-frame=$lattice_depth scp:$dir/IOS+Recorder.lat.JOB.scp ark:- |" \
      "ark:lattice-limit-depth --acoustic-scale=$scale --max-arcs-per-frame=$lattice_depth scp:$dir/Android.lat.JOB.scp ark:- |" ark:- \| \
    lattice-copy-backoff scp:$dir/IOS+Recorder.lat.JOB.scp ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi
if [ $stage -le 2 ];then
  touch $data/text
  local/score.sh --cmd "run.pl" $scoring_opts $data $lang_or_graph $dir
fi
if [ $stage -le 3 ];then
  echo "id,words" >  $dir/submit.csv
  cat $dir/scoring_kaldi/penalty_$wip/$ac_scale.txt | \
  awk '{printf $1",";for(i=2;i<=NF;i++){printf $i};print ""}' | \
  sort -k1,1 | sed 's#<UNK>##g' | sed 's#,$#, #g' | awk -F',' '{print $1","tolower($2)}' >> $dir/submit.csv
fi
