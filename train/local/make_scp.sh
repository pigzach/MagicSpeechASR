#!/bin/bash
if [ $# -ne 2 ];then
  echo "Usage: $0 <src> <scp-dir>"
  exit 1;
fi

src=$1
dest=$2
export LC_ALL=en_US.utf-8

for x in train dev test;do
  mkdir -p $dest/$x
  find $src/audio/$x -iname *.wav | python3 -c "
import sys
for line in sys.stdin:
  uttid = line.strip().split('/')[-1][:-4]
  sys.stdout.write('%s %s\n'%(uttid, line.strip()))
" > $dest/$x/wav.scp
done
find $src/transcription/train -iname *.json | python3 -c "
import sys
import json
segment = {}
text={}
utt2spk={}
utt2location={}
def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = [float(x) for x in timestamp.split(':')]
    return seconds + 60 * (minutes + 60 * hours)
    
for line in sys.stdin:
  json_handler = open(line.strip(),'r')
  info = json.load(json_handler)
  json_handler.close()
  for index,seg in enumerate(info):
    sess = seg['session_id']
    spker = seg['speaker']
    if spker == '': continue
    segid = '%s-%s-%03d'%(spker, sess, index)
    assert(segid not in segment)
    start_time = timestamp_to_seconds(seg['start_time']['original'])
    end_time = timestamp_to_seconds(seg['end_time']['original'])
    segment[segid] = '%s %f %f'%(sess, start_time, end_time)
    text[segid] = seg['words']
    utt2spk[segid] = spker
    utt2location[segid] = seg['location']
with open('$dest/train/segments', 'wt') as f:
  for key, value in segment.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/train/text', 'wt') as f:
  for key, value in text.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/train/utt2spk', 'wt') as f:
  for key, value in utt2spk.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/train/utt2location', 'wt') as f:
  for key, value in utt2location.items():
    f.write('%s %s\n'%(key, value))
"

find $src/transcription/dev -iname *.json | python3 -c "
import sys
import json
segment = {}
text={}
utt2spk={}
utt2location={}
def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = [float(x) for x in timestamp.split(':')]
    return seconds + 60 * (minutes + 60 * hours)
    
for line in sys.stdin:
  json_handler = open(line.strip(),'r')
  info = json.load(json_handler)
  json_handler.close()
  for index,seg in enumerate(info):
    sess = seg['session_id']
    spker = seg['speaker']
    if spker == '': continue;
    start_time = timestamp_to_seconds(seg['start_time']['original'])
    end_time = timestamp_to_seconds(seg['end_time']['original'])
    for device in ['Android', 'IOS', 'Recorder']:
      segid = '%s-%s_%s-%03d'%(spker, sess, device, index)
      uttid = '%s_%s'%(sess, device)
      assert(segid not in segment)
      segment[segid] = '%s %f %f'%(uttid, start_time, end_time)
      text[segid] = seg['words']
      utt2spk[segid] = spker
      utt2location[segid] = seg['location']
    segid = '%s-%s_Clean-%03d'%(spker, sess, index)
    uttid = '%s_%s'%(sess, spker)
    segment[segid] = '%s %f %f'%(uttid, start_time, end_time)
    text[segid] = seg['words']
    utt2spk[segid] = spker
    utt2location[segid] = seg['location']

with open('$dest/dev/segments', 'wt') as f:
  for key, value in segment.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/dev/text', 'wt') as f:
  for key, value in text.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/dev/utt2spk', 'wt') as f:
  for key, value in utt2spk.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/dev/utt2location', 'wt') as f:
  for key, value in utt2location.items():
    f.write('%s %s\n'%(key, value))
"

find $src/transcription/test_no_ref_noise -iname *.json | python3 -c "
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
      # 跳过已经标注为噪声的uttid
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
with open('$dest/test/segments', 'wt') as f:
  for key, value in segment.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/test/utt2spk', 'wt') as f:
  for key, value in utt2spk.items():
    f.write('%s %s\n'%(key, value))
with open('$dest/test/noise.txt', 'wt') as f:
  for key,value in noise_labeled.items():
    f.write('%s,%s\n'%(key, value))
"

for x in train dev test;do
  utils/utt2spk_to_spk2utt.pl < $dest/$x/utt2spk > $dest/$x/spk2utt || exit 1;
  utils/fix_data_dir.sh $dest/$x || exit 1;
done
exit 0;