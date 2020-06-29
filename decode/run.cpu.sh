#!/bin/bash
cmd='run.pl'
stage=0
use_gpu=false
set -e
[ -f ./path.sh ] && . ./path.sh

# setting environment
chmod -R 755 local/*
chmod 755 *.sh
ln -sf $KALDI_ROOT/egs/wsj/s5/{utils,steps,rnnlm} ./

. utils/parse_options.sh

if [ $# -ne 1 ];then
  echo "Usage: $0 <test-data>"
  exit 1;
fi
test_corps=$1

start_tm=`date +%s%N`
echo '[Stage 0: Prepare test data ...]'
if [ $stage -le 0 ];then
  local/make_scp.sh $test_corps $test_corps data/mfcc_test
fi

nj=$(cat data/mfcc_test/spk2utt |wc -l)
echo '[Stage 1: Extract mfcc+pitch and ivector ...]'
if [ $stage -le 1 ];then
  steps/make_mfcc_pitch.sh --compress false --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
      --cmd "$cmd" --nj $nj data/mfcc_test || exit 1;
  steps/compute_cmvn_stats.sh data/mfcc_test || exit 1;
  utils/fix_data_dir.sh data/mfcc_test || exit 1;
  # create MFCC data dir without pitch to extract iVector
  utils/data/limit_feature_dim.sh 0:39 data/mfcc_test data/mfcc_test_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/mfcc_test_nopitch || exit 1;
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 20 \
    data/mfcc_test_nopitch exp/ivector/extractor data/ivector_test || exit 1;
fi

echo '[Stage 2: Run decode with language model ...]'
if [ $stage -le 2 ];then
  if $use_gpu;then
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 2 --cmd "$cmd" --num-threads $nj --use-gpu $use_gpu \
      --online-ivector-dir data/ivector_test \
      --skip-diagnostics true --skip-scoring true \
      exp/chain/graph data/mfcc_test exp/chain/cnn-tdnnf/decode_test_gpu
    local/split_lattice.sh exp/chain/cnn-tdnnf/decode_test_gpu $nj exp/chain/cnn-tdnnf/decode_test
  else 
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$cmd" \
      --online-ivector-dir data/ivector_test \
      --skip-diagnostics true --skip-scoring true \
      exp/chain/graph data/mfcc_test exp/chain/cnn-tdnnf/decode_test
  fi
fi

echo '[Stage 3: Run rnnlm rescoring forward ...]'
if [ $stage -le 3 ];then
  rnnlm/lmrescore_pruned.sh \
    --cmd "$cmd" \
    --weight 0.45 --max-ngram-order 4 --skip-scoring true \
    exp/chain/lang exp/rnnlm/forward \
    data/mfcc_test exp/chain/cnn-tdnnf/decode_test \
    exp/rnnlm/forward/decode_test_rnnlm
fi

echo '[Stage 4: Run rnnlm rescoring backward ...]'
if [ $stage -le 4 ];then
  rnnlm/lmrescore_back.sh \
    --cmd "$cmd" \
    --weight 0.3 --max-ngram-order 4 --skip-scoring true \
    exp/chain/lang exp/rnnlm/backward \
    data/mfcc_test exp/rnnlm/forward/decode_test_rnnlm \
    exp/rnnlm/backward/decode_test_rnnlm_back
fi

echo '[Stage 5: Combine channel data and generate final submit result ...]'
if [ $stage -le 5 ];then
  local/combine_lattice_and_generate_submit_limit_depth.sh --cmd "$cmd" \
    --wip 1.0 --ac-scale 11 --lattice-depth 150 \
    --scoring-opts "--word-ins-penalty 1.0 --min-lmwt 10 --max-lmwt 12  --decode-mbr true" \
    data/mfcc_test exp/chain/lang exp/rnnlm/backward/decode_test_rnnlm_back exp/final-submit
fi

end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}'`
echo "Done, total time cost: $use_tm seconds."
