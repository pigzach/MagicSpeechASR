#!/bin/bash
stage=0
train_nj=64
dict=conf/lexicon.txt

set -e
[ -f ./cmd.sh ] && . ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh
if [ $# -ne 1 ];then
  echo "$0 <data-base>"
  exit 1;
fi

corps=$1

echo "[ Stage 0] run.sh: Clone Text-norm ..."
if [ $stage -le 0 ];then
  git clone https://github.com/speech-io/chinese_text_normalization.git
fi

echo "[ Stage 1 ] run.sh: Prepare lexicon ..."
if [ $stage -le 1 ];then
  local/prepare_dict.sh $dict data/local/dict
  # Phone Sets, questions, L compilation
  utils/prepare_lang.sh --position-dependent-phones true data/local/dict \
    "<UNK>" data/local/lang data/lang || exit 1;
fi

echo "[ Stage 2 ] run.sh: Prepare data ..."
if [ $stage -le 2 ];then
  local/make_scp.sh $corps data/tmp || exit 1;
  local/prepare_data.sh data/tmp/train data/local/dict data/local/train data/train || exit 1;
  local/prepare_data.sh data/tmp/dev data/local/dict data/local/dev data/dev || exit 1;
  local/prepare_data.sh data/tmp/test data/local/dict data/local/test data/test || exit 1;
  rm -rf data/tmp
fi

# nj for dev and test
dev_nj=$(wc -l data/dev/spk2utt | awk '{print $1}' || exit 1;)
test_nj=$(wc -l data/test/spk2utt | awk '{print $1}' || exit 1;)
# Now make MFCC features.
echo "[ Stage 3 ] run.sh: Make mfcc feature ..."
if [ $stage -le 3 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  for x in train dev; do
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf \
      --cmd "$train_cmd" --nj $train_nj \
      data/$x exp/make_mfcc/$x mfcc || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
fi

echo "[ Stage 4 ] run.sh: Subset training data dir ..."
if [ $stage -le 4 ]; then
  # subset the training data for fast startup
  for x in 50 100; do
    utils/subset_data_dir.sh data/train ${x}000 data/train_${x}k
  done
fi

# mono
echo "[ Stage 5 ] run.sh: Monophone training ..."
if [ $stage -le 5 ]; then
  # training
  steps/train_mono.sh --cmd "$train_cmd" --nj $train_nj \
    data/train_50k data/lang exp/mono || exit 1;

  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train_100k data/lang exp/mono exp/mono_ali || exit 1;
fi

# tri1
echo "[ Stage 6 ] run.sh: Triphone training ..."
if [ $stage -le 6 ]; then
  # training
  steps/train_deltas.sh --cmd "$train_cmd" \
   4000 32000 data/train_100k data/lang exp/mono_ali exp/tri1 || exit 1;

  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
fi

# tri2
echo "[ Stage 7 ] run.sh: Triphone training (more data) ..."
if [ $stage -le 7 ]; then
  # training
  steps/train_deltas.sh --cmd "$train_cmd" \
   7000 56000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
fi

# tri3
echo "[ Stage 8 ] run.sh: Triphone lda+mllt training ..."
if [ $stage -le 8 ]; then
  # training [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
   10000 80000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1;

  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $train_nj \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1;
fi


# tri4
echo "[ Stage 9 ] run.sh: Triphone sat+fmllr training ..."
if [ $stage -le 9 ];then
  steps/train_sat.sh  --cmd "$train_cmd" \
                      11500 120000 data/train data/lang exp/tri3_ali exp/tri4
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $train_nj \
                       data/train data/lang exp/tri4 exp/tri4_ali
fi

# tri5
echo "[ Stage 10 ] run.sh: Triphone sat+fmllr training(more gaussian and leaves) ..."
if [ $stage -le 10 ];then
  steps/train_sat.sh  --cmd "$train_cmd" \
                      12000 200000 data/train data/lang exp/tri4_ali exp/tri5
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $train_nj \
                       data/train data/lang exp/tri5 exp/tri5_ali
fi

echo "[ Stage 11 ] run.sh: Adjust lexicon probabilities ..."
if [ $stage -le 11 ]; then
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang exp/tri5
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict exp/tri5/pron_counts_nowb.txt exp/tri5/sil_counts_nowb.txt \
                                  exp/tri5/pron_bigram_counts_nowb.txt data/local/dict_sp

  utils/prepare_lang.sh --position-dependent-phones true data/local/dict_sp "<UNK>" data/local/lang_sp data/lang_sp
fi

echo "[ Stage 12 ] run.sh: Train language model and test our GMM model on dev set ..."
if [ $stage -le 12 ];then
  local/train_lms.sh \
      data/local/dict/lexicon.txt \
      data/train/text \
      data/local/lm || exit 1;
  # G compilation, check LG composition
  utils/format_lm.sh data/lang_sp data/local/lm/4gram-mincount/lm_unpruned.gz \
    data/local/dict_sp/lexicon.txt data/lang_test_sp || exit 1;
  utils/mkgraph.sh data/lang_test_sp exp/tri5 exp/tri5/graph_sp
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$decode_cmd" \
                        exp/tri5/graph_sp data/dev exp/tri5/decode_dev_sp
fi

echo "[ Stage 13 ] run.sh: Blance the data of training set and dev set, augment data by perturb speed and volume ..."
if [ $stage -le 13 ];then
  local/train_data_copyx3.sh data/train data/train_copyx3
  utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true data/train data/train_speed_hires
  utils/data/perturb_data_dir_volume.sh data/train_speed_hires
  utils/combine_data.sh data/train_aug data/train_copyx3 data/train_speed_hires

  utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true data/dev data/dev_aug
  utils/data/perturb_data_dir_volume.sh data/dev_aug
fi

echo "[ Stage 14 ] run.sh: Extract 40dims mfcc + 3dim pithc on the augment data ..."
if [ $stage -le 14 ];then
  for x in train_aug dev_aug test; do
    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
      --cmd "$train_cmd" --nj $train_nj \
      data/$x exp/make_mfcc_hires/$x mfcc_hires || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc_hires/$x mfcc_hires || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done

  utils/combine_data.sh data/train_dev_aug data/train_aug data/dev_aug
  # create MFCC data dir without pitch to extract iVector
  utils/data/limit_feature_dim.sh 0:39 data/train_dev_aug data/train_dev_aug_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/train_dev_aug_nopitch exp/make_mfcc_hires/train_dev_aug_nopitch mfcc_hires || exit 1;

  utils/data/limit_feature_dim.sh 0:39 data/test data/test_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/test_nopitch exp/make_mfcc_hires/test_nopitch mfcc_hires || exit 1;

  cp -r data/train_dev_aug data/train_dev_aug_16dims
  rm -rf data/train_dev_aug_16dims/{feats.scp,cmvn.scp}
  steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf \
      --cmd "$train_cmd" --nj $train_nj \
      data/train_dev_aug_16dims exp/make_mfcc_hires/train_dev_aug_16dims mfcc_hires || exit 1;
  steps/compute_cmvn_stats.sh data/train_dev_aug_16dims exp/make_mfcc_hires/train_dev_aug_16dims mfcc_hires || exit 1;
  utils/fix_data_dir.sh data/train_dev_aug_16dims || exit 1;
fi

echo "[ Stage 15 ] run.sh: Train ivector extractor and extract ivector of the training set, dev set and test set ..."
if [ $stage -le 15 ];then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  mkdir -p exp/ivector/diag_ubm
  temp_data_root=exp/ivector/diag_ubm

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
      data/train_dev_aug_nopitch exp/ivector/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $train_nj \
    --num-frames 700000 \
    --num-threads 32 \
    data/train_dev_aug_nopitch 512 \
    exp/ivector/pca_transform exp/ivector/diag_ubm

  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $train_nj \
     data/train_dev_aug_nopitch exp/ivector/diag_ubm \
     exp/ivector/extractor || exit 1;

  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/train_dev_aug_nopitch data/train_dev_aug_nopitch_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $train_nj \
    data/train_dev_aug_nopitch_max2 \
    exp/ivector/extractor exp/ivector/ivector_train_dev_aug

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $test_nj \
      data/test_nopitch exp/ivector/extractor \
      exp/ivector/ivector_test

fi

echo "[ Stage 16 ] run.sh: Train nnet3 CE model on the train+dev augment data set, for better alignment  ..."
if [ $stage -le 16 ];then
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $train_nj \
                       data/train_dev_aug_16dims data/lang_sp exp/tri5 exp/tri5_ali_train_dev_aug
  echo "$0: creating neural net configs";

  train_data_dir=data/train_dev_aug
  train_ivector_dir=exp/ivector/ivector_train_dev_aug
  ali_dir=exp/tri5_ali_train_dev_aug
  dir=exp/nnet3/tdnn

  train_nnet=true
  if $train_nnet;then
    num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')
    opts="l2-regularize=0.002"
    output_opts="l2-regularize=0.0005 bottleneck-dim=256"

    mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 dim=1024 $opts
  relu-batchnorm-dropout-layer name=tdnn2 dim=1024 input=Append(-1,0,2) $opts
  relu-batchnorm-dropout-layer name=tdnn3 dim=1024 input=Append(-3,0,3) $opts
  relu-batchnorm-dropout-layer name=tdnn4 dim=1024 input=Append(-7,0,2) $opts
  relu-batchnorm-dropout-layer name=tdnn5 dim=1024 input=Append(-3,0,3) $opts
  relu-batchnorm-dropout-layer name=tdnn6 dim=1024 $opts
  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5 $output_opts

EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
    steps/nnet3/train_dnn.py --stage -10 \
      --cmd="$cuda_cmd" \
      --feat.online-ivector-dir $train_ivector_dir \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --trainer.num-epochs 4 \
      --trainer.optimization.num-jobs-initial 2 \
      --trainer.optimization.num-jobs-final 12 \
      --trainer.optimization.initial-effective-lrate 0.0015 \
      --trainer.optimization.final-effective-lrate 0.00015 \
      --cleanup.remove-egs true \
      --use-gpu true \
      --cleanup.preserve-model-interval 100 \
      --feat-dir=$train_data_dir \
      --ali-dir $ali_dir \
      --lang data/lang_sp \
      --dir=$dir  || exit 1;
  fi

  steps/nnet3/align_lats.sh --cmd "$train_cmd" --nj $train_nj --generate-ali-from-lats true \
    --online-ivector-dir $train_ivector_dir $train_data_dir data/lang_sp $dir ${dir}_ali

fi

echo "[ Stage 17 ] run.sh: Train nnet3 chain model on the train+dev augment data set ..."
if [ $stage -le 17 ];then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  train_data_dir=data/train_dev_aug
  train_ivector_dir=exp/ivector/ivector_train_dev_aug

  lang=exp/chain/lang
  treedir=exp/chain/tree
  graphdir=exp/chain/graph
  latdir=exp/nnet3/tdnn_ali
  dir=exp/chain/cnn-tdnnf

  train_stage=-10
  get_egs_stage=-10
  xent_regularize=0.1

  mkdir -p exp/chain
  cp -r data/lang_sp $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo

  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 $train_data_dir $lang $latdir $treedir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.01"
  tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # MFCC to filterbank
  dim-range-component name=mfcc input=input dim=40 dim-offset=0
  dim-range-component name=pitch input=input dim=3 dim-offset=40
  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=mfcc dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  linear-component name=ivector-linear l2-regularize=0.01 dim=197 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct
  batchnorm-component name=pitch-batchnorm input=pitch
  spec-augment-layer name=idct-spec-augment input=idct-batchnorm freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20

  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm, pitch-batchnorm) num-filters1=1 num-filters2=5 height=40
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-3,-2,-1,0,1,2,3 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10  time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.  (we use time-stride=0 so no splicing, to
  # limit the num-parameters).
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf19 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf20 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf21 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf22 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf23 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts
  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule "0,0@0.20,0.3@0.50,0" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width "150,110,90,30" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.00015 \
    --trainer.optimization.final-effective-lrate 0.000015 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $latdir \
    --dir $dir  || exit 1;

  mkdir -p data/local/lm_train_dev
  cat data/train/text <(cat data/dev/text | grep Android) >  data/local/lm_train_dev/text
  local/train_lms.sh \
      data/local/dict_sp/lexicon.txt \
      data/local/lm_train_dev/text \
      data/local/lm_train_dev || exit 1;

  utils/format_lm.sh $lang data/local/lm_train_dev/4gram-mincount/lm_unpruned.gz \
    data/local/dict_sp/lexicon.txt exp/chain/lang_test || exit 1;
  utils/mkgraph.sh --self-loop-scale 1.0 exp/chain/lang_test $treedir $graphdir
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $test_nj --cmd "$decode_cmd" \
    --online-ivector-dir exp/ivector/ivector_test \
    $graphdir data/test \
    $dir/decode_test
fi

echo "[ Stage 18 ] run.sh: Fine tuning the chain model with dev_aug data ..."
dir=exp/chain/cnn-tdnnf-finetune
if [ $stage -le 18 ];then
  mkdir -p $dir
  $train_cmd $dir/log/generate_input_model.log \
    nnet3-am-copy --raw=true $src_dir/final.mdl $dir/input.raw
    
  utils/data/limit_feature_dim.sh 0:39 data/dev_aug data/dev_aug_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/dev_aug_nopitch exp/make_mfcc_hires/dev_aug_nopitch mfcc_hires || exit 1;
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $train_nj \
        data/dev_aug_nopitch exp/ivector/extractor \
        exp/ivector/ivector_dev_aug

  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  
  train_data_dir=data/dev_aug
  train_ivector_dir=exp/ivector/ivector_dev_aug

  lang=exp/chain/lang
  treedir=exp/chain/tree
  latdir=exp/nnet3/tdnn_ali_dev_aug
  
  if ! [ -d $latdir ];then
    steps/nnet3/align_lats.sh --cmd "$train_cmd" --nj $train_nj --generate-ali-from-lats true \
      --online-ivector-dir $train_ivector_dir $train_data_dir data/lang_sp exp/nnet3/tdnn $latdir
  fi

  train_stage=-10
  get_egs_stage=-10
  xent_regularize=0.1

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.input-model $dir/input.raw \
    --trainer.dropout-schedule "0,0@0.20,0.3@0.50,0" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width "150,110,90,30" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 8 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.000015 \
    --trainer.optimization.final-effective-lrate 0.000010 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $latdir \
    --dir $dir  || exit 1;

  graphdir=exp/chain/graph
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $test_nj --cmd "$decode_cmd" \
    --online-ivector-dir exp/ivector/ivector_test \
    $graphdir data/test \
    $dir/decode_test
fi

echo "[ Stage 19 ] run.sh: Train forward rnnlm model with pretrain Tencent word embedding and rescoring the fine-tune chain model decode result ..."
if [ $stage -le 19 ];then
  tencent_embedding=data/wordembedding/Tencent_AILab_ChineseEmbedding.txt
  if ! [ -e $tencent_embedding ];then
    echo "Can not find $tencent_embedding, please download it from: https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz"
    echo "tar -zxvf Tencent_AILab_ChineseEmbedding.tar.gz and copy the Tencent_AILab_ChineseEmbedding.txt to data/wordembedding"
    echo ", after that, re-run this scripts as: $0 --stage 21."
  fi
  local/train_rnnlm_with_pretrain_embedding.sh 
  rnnlm/lmrescore_pruned.sh \
    --cmd "$decode_cmd" \
    --weight 0.45 --max-ngram-order 4 \
    exp/chain/lang_test exp/rnnlm_lstm_tdnn_pretrain \
    data/test exp/chain/cnn-tdnnf-finetune/decode_test \
    exp/rnnlm_lstm_tdnn_pretrain/decode_test

fi

echo "[ Stage 20 ] run.sh: Train backward rnnlm model with pretrain Tencent word embedding and rescoring the forward rnnlm decode result ..."
if [ $stage -le 20 ];then
  local/train_rnnlm_with_pretrain_embedding_back.sh
  rnnlm/lmrescore_back.sh \
    --cmd "$decode_cmd" \
    --weight 0.3 --max-ngram-order 4 \
    exp/chain/lang_test exp/rnnlm_lstm_tdnn_pretrain_back \
    data/test exp/rnnlm_lstm_tdnn_pretrain/decode_test \
    exp/rnnlm_lstm_tdnn_pretrain_back/decode_test
fi

if [ $stage -le 23 ];then
  local/combine_lattice_and_generate_submit_limit_depth.sh --cmd "$decode_cmd" \
    --wip 1.0 --ac-scale 11 --lattice-depth 150 \
    --scoring-opts "--word-ins-penalty 1.0 --min-lmwt 10 --max-lmwt 12  --decode-mbr true" \
    data/test exp/chain/lang_test exp/rnnlm_lstm_tdnn_pretrain_back/decode_test exp/final-submit
fi

echo "Finish."
exit 0;






