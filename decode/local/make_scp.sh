#!/bin/bash
if [ $# -ne 3 ];then
  echo "Usage: $0 <src-audio> <src-transcript> <scp-dir>"
  exit 1;
fi

audio=$1
trans=$2
dest=$3

mkdir -p $dest
export LC_ALL=en_US.utf-8
find $audio -iname *.wav | grep 'Android\|IOS\|Recorder' | grep -v '\._' | python3 -c "
import sys
for line in sys.stdin:
  uttid = line.strip().split('/')[-1][:-4]
  sys.stdout.write('%s %s\n'%(uttid, line.strip()))
" > $dest/wav.scp

find $trans -iname *.json | grep -v '\._' | python3 -c "
import sys
import json
segment = {}
utt2spk = {}
noise_labeled={}

def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = [float(x) for x in timestamp.split(':')]
    return seconds + 60 * (minutes + 60 * hours)

for line in sys.stdin:
  json_handler = open(line.strip(),'r')
  info = json.loads(json_handler.read().replace('\n','').replace(' ','').replace(',}','}'))
  json_handler.close()
  for index,seg in enumerate(info):
    if 'words' in seg:
      noise_labeled[seg['uttid']] = seg['words']
      continue
      
    for device in ['Android', 'IOS', 'Recorder']:
      sess = seg['session_id']
      segid = seg['uttid'].replace('-','_%s-'%device)
      uttid = '%s_%s'%(sess, device)
      assert(segid not in segment)
      start_time = timestamp_to_seconds(seg['start_time'])
      end_time = timestamp_to_seconds(seg['end_time'])
      segment[segid] = '%s %f %f'%(uttid, start_time, end_time)
      utt2spk[segid] = uttid
with open('$dest/segments', 'wt') as f:
  for key, value in segment.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/utt2spk', 'wt') as f:
  for key, value in utt2spk.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/noise.txt', 'wt') as f:
  for key,value in noise_labeled.items():
    f.write('%s,%s\n'%(key, value))
"
utils/utt2spk_to_spk2utt.pl < $dest/utt2spk > $dest/spk2utt
utils/fix_data_dir.sh $dest

echo "$0: finish prepare data."