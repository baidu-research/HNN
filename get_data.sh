#!/bin/bash
mkdir -p data/dictionaries
mkdir -p data/embeddings

# download dictionaries
for src_lg in de en es fr it pt
do
  for tgt_lg in de en es fr it pt
  do
    if [ $src_lg != $tgt_lg ]
    then
      for suffix in .txt .0-5000.txt .5000-6500.txt
      do
        fname=$src_lg-$tgt_lg$suffix
        curl -Lo data/dictionaries/$fname https://dl.fbaipublicfiles.com/arrival/dictionaries/$fname
      done
    fi
  done
done

# download embeddings
for lg in de en es fr it pt
do
  curl -Lo data/embeddings/wiki.$lg.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$lg.vec
done
