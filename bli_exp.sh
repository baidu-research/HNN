#!/bin/bash
V=500000;
V_str=$((V/1000))K;
langs="de en es fr it pt";
emb_dir=data/embeddings;
dict_dir=data/dictionaries;
exp_dir=exp/bli_$V_str;
mkdir -p $exp_dir;
for src in $langs
  do for tgt in $langs
    do if [ $src != $tgt ]
       then
        for mtd in nn isf csls hnn
        do sbatch --wrap "python bilingual_lexicon_induction.py ${emb_dir}/wiki.${src}.vec ${emb_dir}/wiki.${tgt}.vec -m ${V} -n ${V} -d ${dict_dir}/${src}-${tgt}.0-5000.txt -t ${dict_dir}/${src}-${tgt}.txt --orth --method ${mtd}" -o ${exp_dir}/${src}-${tgt}.${mtd}.log -p TitanXx8 --gres=gpu:1;
        done;
       fi;
  done;
done;
