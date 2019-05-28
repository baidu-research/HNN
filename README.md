# HNN
Hubless Nearest Neighbor Search

Neareast Neighbor (NN) Search is widely applied in retrieval tasks. However, a phenomenon called hubness [[1]](http://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) often degrades NN's performance.
Hubness appears as some data points, called hubs, being suspiciously close to many others. It is often encountered in high dimensional spaces.

This work is interested in reducing hubness during retrieval. The proposed is a new method, which we call Hubless Nearest Neighbor (HNN) Search.
Theoretically, HNN provides a unified view towards NN and Inverted SoFtmax (ISF [[2]](https://arxiv.org/pdf/1702.03859.pdf)), a recently proposed retrieval method that mitigates hubness.
Empirically, HNN demonstrates superior accuracy in a typical retrieval task, Bilingual Lexicon Induction (BLI [[3]](https://arxiv.org/pdf/1710.04087.pdf)).

The repo relies on several packages, including `numpy`, `scipy`, `gensim`, `cupy` (optional, only if cuda available). For easiness, simply run `source activate.sh` to use an existing virtual environment.

Run `sbatch -p TitanXx8 --gres=gpu:1 --wrap "python GMM_retrieval.py" -o ./exp/GMM_retrieval.out` to see a synthetic example. The task is simply to retrieve the same class from a Gaussian mixture. An examplar output is given in `./exp/GMM_retrieval.out`.

Run `./bli_exp.sh` to get results of BLI. The experiment follows the "supervised" setup at [MUSE](https://github.com/facebookresearch/MUSE) and uses the same sets of word embeddings. The outputs are logs (`src-tgt.method.log`) and translated words (`src-tgt.method.txt`) stored under `./exp/bli_500K`, where 500K is the vocabulary size for both source and target languages.

After the jobs are done, we can check how hubness is reduced. For example, to check the hubness for Portuguese-to-English task, simply run
```
python hubness_in_translations.py pt en -k 5 -N 200
```
It will produce `k-occurrence` (k=5 in this case) histograms, as measures of hubness, for the different methods. In particular, long tail of the histogram indicates strong hubness, which should be reduced. The Portuguese to English example will have the following histograms, where HNN has the shortest tail, *i.e.*, weakest hubness.
<p align="center">
    <img src="exp/bli_500K/pt-en.k_occur.png" width="400">
</p>
We will also see some "hubby" words being listed, for example:


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
