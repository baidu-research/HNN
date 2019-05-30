![Baidu Logo](/doc/baidu-research-logo-small.png)

- [HNN](#HNN)
- [Prerequisites](#Prerequisites)
- [Experiments](#Experiments)

# HNN
This repositorty contains the code to reproduce major results from the paper:

Jiaji Huang, Qiang Qiu and Kenneth Church. Hubless Nearest Neighbor Search for Bilingual Lexicon Induction. ACL 2019

Neareast Neighbor (NN) Search is widely applied in retrieval tasks. However, a phenomenon called hubness [[1]](http://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) often degrades NN's performance.
Hubness appears as some data points, called hubs, being suspiciously close to many others. It is often encountered in high dimensional spaces.

This work is interested in reducing hubness during retrieval. The proposed is a new method, which we call Hubless Nearest Neighbor (HNN) Search.
Theoretically, HNN provides a unified view towards NN and Inverted SoFtmax (ISF [[2]](https://arxiv.org/pdf/1702.03859.pdf)), a recently proposed retrieval method that mitigates hubness.
Empirically, HNN demonstrates superior accuracy in a typical retrieval task, Bilingual Lexicon Induction (BLI [[3]](https://arxiv.org/pdf/1710.04087.pdf)).

If you have any question, please post it on github or email authentichj@outlook.com

# Prerequisites
Environment
* python (3.6.6)

Mandatory Packages
* numpy (1.15.4)

* scipy (1.1.0)

* matplotlib (3.0.2)

* gensim (3.6.0)

All of the above can be installed via `pip`, e.g.,
```
pip install 'numpy==1.15.4'
```

Optional Packages (if use GPU)
* cupy-cuda90

Assume cuda available version is 9.0. Install it by
```
pip install cupy-cuda90
```
Also, append the CUDA paths in bash environment. The following is a working example:
```
CUDA_PATH=/path/to/cuda-9.0.176
CUDNN_PATH=/path/to/cudnn-9.0-linux-x64-v7.0-rc
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:$CUDNN_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH
```
Note: Other versions of Python and the above packages are not tested. But we believe they should work as long as python3+ is used.

# Experiments
## Synthetic Data
The following will reproduce Table 1 of the paper. Simply run
```
python GMM_retrieval.py
```
The task is to retrieve the same class from a Gaussian mixture. Details can be found in section 5.1 of the paper. The purpose of this experiment is to understand the connection between HNN and other related methods. An example output is already given in `./exp/GMM_retrieval.example.log`.

The log includes the accuracies of different retrieval methods, as well as evidence that primal and dual solvers of HNN are equivalent. Results in Table 1 can be easily found in the log.

As we discussed in the paper, dual solver of HNN minimizes a convex objective. Therefore, if the learning rate in algorithm 2 is properly chosen, the loss and gradient norm should be monotonicly decreasing. This fact can be easily checked by parsing the log file (assuming `gnuplot` installed)
```
grep "grad norm" exp/GMM_retrieval.example.log | cut -d, -f2 | cut -d: -f2 | gnuplot -p -e "set xlabel 'iteration'; set ylabel 'gradient norm'; plot '-' with line notitle;"
```
This produces a plot of the gradient norms over the iterations in algorithm 2.
<p align="center">
    <img src="doc/gradient_norm.png" width="400">
</p>

## Bilingual Lexicon Induction
The following will reproduce Table 3 of the paper.

(1) Download the fasttext embeddings and dictionaries.
```
./get_data.sh
```
A ./data directory will be created. Under that are embeddings for 6 European languages, de (German), en (English), es (Spanish), fr (French), it (Italian) and pt (Portuguese), as well as dictionaries for all the pairs.

(2) Get translation accuracy for a `src`-`tgt` pair, using a specified retrieval `method` (one of {nn, isf, csls, hnn}). Run
```
./bli_exp.sh $src $tgt $method
```
The experiment follows the "supervised" setup at [MUSE](https://github.com/facebookresearch/MUSE), but differs in that we use larger test dictionaries (data/`src`-`tgt`.txt). The output is a log, `/exp/BLI/src-tgt.method.log`. 

By default, we use 500K as the vocabulary size for both source and target languages. In supplementary material, We have also reported results of using a vocabulary of 200K. To reproduce that, simply change `V=200000` in `bli_exp.sh`.

To see all translated words, set `save_translation=1` in `bli_exp.sh`. The translated words will be in `./exp/BLI/src-tgt.method.trans`. Each row of the file is a source word, followed by 10 top candidates of translation, from the "best" to the "worst".

(3) Check how hubness is reduced (Figure 4 and Table 2). For example, to check the hubness for Portuguese-to-English task, simply run
```
python hubness_in_translations.py pt en -k 5 -N 200
```
It will produce `k-occurrence` (k=5 in this case) histograms, as measures of hubness, for the different methods. In particular, long tail of the histogram indicates strong hubness, which should be reduced. The Portuguese to English example will have the following histograms, where HNN has the shortest tail, *i.e.*, weakest hubness.
<p align="center">
    <img src="doc/pt-en.k_occur.png" width="400">
</p>
We will also see some (200 in this case) "hubby" words being listed, for example:

| "hubby" words |   NN  | ISF | CSLS | HNN |
|:-------------:|:-----:|:---:|:----:|:---:|
|   conspersus  | 1,776 |   0 |  374 |   0 |
|      s+bd     |   912 |   7 |  278 |  16 |
|      were     |   798 |  99 |  262 |  24 |
|      you      |   474 |  12 |   57 |  20 |

The numerics are the number of times these words being retrieved. A big value indicates that the word is a hub. Note how the values are reduced by HNN.

[[1]](http://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) Milos Radovanovic, Alexandros Nanopoulos, and Mirjana Ivanovic. 2010. Hubs in space: Popular nearest neighbors in high-dimensional data. Journal of Machine Learning Research.

[[2]](https://arxiv.org/pdf/1702.03859.pdf) Samuel L. Smith, David H. P. Turban, Steven Hamblin, and Nils Y. Hammerla. 2017. Offline bilingual word vectors, orthogonal transformations and the inverted softmax. In International Conference on Learning Representations.

[[3]](https://arxiv.org/pdf/1710.04087.pdf) Alexis Conneau, Guillaume Lample, Marc' Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In International Conference on Learning Representations.
