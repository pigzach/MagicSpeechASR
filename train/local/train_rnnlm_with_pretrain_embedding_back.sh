#!/usr/bin/env bash
# Begin configuration section.

dir=exp/rnnlm_lstm_tdnn_pretrain_back
lstm_rpd=100
lstm_nrpd=100
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=50
stage=-10
train_stage=-10

. ./cmd.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd


text_train=data/train/text
text_dev=data/dev/text
wordlist=data/lang_sp/words.txt
text_dir=data/rnnlm_pretrain/text_back
src_embedding=data/wordembedding/Tencent_AILab_ChineseEmbedding.txt
mkdir -p $dir/config
set -e

for f in $text_train $text_dev $wordlist; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/magic_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 500 lines as dev data.
  cat $text_train <(cat $text_dev | grep Android) | tr '\t' ' ' | cut -d' ' -f2- | local/filter-oov.py --oov '<UNK>' --word-list conf/wordlist | \
  awk '{for(i=NF;i>0;i--) if(i == 1) {printf("%s", $i); print"";} else {printf("%s ", $i);}}' | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/magic.txt
fi

if [ $stage -le 1 ]; then
  # the training scripts require that <s>, </s> and <brk> be present in a particular
  # order.
  cp $wordlist $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt
  echo "<UNK>" > $dir/config/oov.txt
  cat > $dir/config/data_weights.txt <<EOF
magic   1   1.0
EOF
embedding_dim=$(head -n1 $src_embedding | awk '{print $2}')
lstm_opts="l2-regularize=$comp_l2"
tdnn_opts="l2-regularize=$comp_l2"
output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn2 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-2))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn3 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
output-layer name=output $output_opts include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200.0 \
                             $text_dir $dir/config $dir
  local/extract_wordembedding.sh $src_embedding $dir/config/words.txt $dir/word_embedding.0.mat
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi

exit 0
