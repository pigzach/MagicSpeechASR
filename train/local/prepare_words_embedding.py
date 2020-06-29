#!/usr/bin/env python3
import sys
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

if len(sys.argv) != 3:
  sys.stderr.write("local/prepare_words_embedding.py <src-mdl> <dest-mdl>\n")
  sys.exit(1)

srcmdl=sys.argv[1]
dstmdl=sys.argv[2]

model = KeyedVectors.load_word2vec_format(srcmdl)
annoy_index = AnnoyIndexer(model,200)
annoy_index.save(dstmdl)

