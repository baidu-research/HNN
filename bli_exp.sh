#!/bin/bash
V=500000
save_translation=1
src=$1
tgt=$2 # src and tgt must be from {de, en, es, fr, it, pt}
mtd=$3 # one of {nn, isf, csls, hnn}

V_str=$((V/1000))K
emb_dir=data/embeddings
dict_dir=data/dictionaries
exp_dir=exp/BLI
mkdir -p $exp_dir
if [ $save_translation ]; then
  python bilingual_lexicon_induction.py $emb_dir/wiki.$src.vec $emb_dir/wiki.$tgt.vec -m $V -n $V -d $dict_dir/$src-$tgt.0-5000.txt -t $dict_dir/$src-$tgt.txt --orth --method $mtd --save_translation $exp_dir/$src-$tgt.$mtd.trans > $exp_dir/$src-$tgt.$mtd.log &
else
  python bilingual_lexicon_induction.py $emb_dir/wiki.$src.vec $emb_dir/wiki.$tgt.vec -m $V -n $V -d $dict_dir/$src-$tgt.0-5000.txt -t $dict_dir/$src-$tgt.txt --orth --method $mtd > $exp_dir/$src-$tgt.$mtd.log &
fi
echo "job $! submitted!"
