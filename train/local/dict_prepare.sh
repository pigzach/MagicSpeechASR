#!/bin/bash
oov='<unk>'
echo "$0 $@"
. utils/parse_options.sh
export LC_ALL=en_US.UTF-8
if [ $# -ne 2 ];then
  echo "Usage: "
  echo "    $0 dict dir"
  echo "    e.g. $0 database/lexicon.txt data/dict"
  exit 1
fi

dict=$1
dir=$2

mkdir -p $dir
cut -d ' ' -f 2- $dir/lexicon.txt | tr ' ' '\n' | sort -u | grep -v 'sil\|spn' > $dir/nonsilence_phones.txt
(
  echo sil
  echo spn
) > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt
cat $dir/silence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

echo "Dict preparation succeeded."
exit 0;
