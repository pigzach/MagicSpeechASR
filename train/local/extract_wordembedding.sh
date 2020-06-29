#!/bin/bash
max_warn=50
if [ $# -ne 3 ];then
  echo "Usage: $0 <src-embedding> <wordlist> <dest-embedding>"
  exit 1;
fi
src_embedding=$1
word_list=$2
dest_embedding=$3

cat $src_embedding | python3 -c "
import sys
import random
word_embedding={}
first_line = True
embedding_dim=0
for line in sys.stdin:
  word, vec = line.strip().split(' ', 1)
  if first_line:
    embedding_dim = int(vec)
    first_line = False
  else :
    word_embedding[word] = vec

num_warn = 0
with open('$word_list', 'r') as f:
  sys.stdout.write('[ ')
  for line in f:
    word = line.strip().split(' ', 1)[0].lower()
    if word in word_embedding:
      sys.stdout.write(word_embedding[word] + '\n')
    else :
        num_warn+=1
        if num_warn <= $max_warn :
          sys.stderr.write('[WARN] random embedding of word: %s.\n'%word)
          sys.stderr.flush()
          if num_warn == $max_warn :
            sys.stderr.write('[WARN] have reached max warning: $max_warn, stop logout warning ...\n')
            sys.stderr.flush()
        sys.stdout.write(' '.join(['%.6f'%(2*random.random()-1) for _ in range(embedding_dim)]) + '\n')
  sys.stdout.write(' ]')
" > $dest_embedding

echo "Done!"

