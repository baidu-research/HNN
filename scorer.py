import os
import sys
from function_utils import *
from io_utils import create_path_for_file
from HNN import HNN
try:
    import cupy as xp
    gpu = True
except ImportError:
    import numpy as xp
    gpu = False


class scorer:
    """ Build a scoring function given the retrieval method (nn, hnn, etc).
        Translate a list of queries
    """
    def __init__(self, linear_map, src_emb, tgt_emb, src_vocab, tgt_vocab):
        self.m = src_emb.shape[0]
        self.n = tgt_emb.shape[0]
        assert len(src_vocab) == self.m
        assert len(tgt_vocab) == self.n
        self.src_space = src_emb.dot(linear_map)
        self.tgt_space = tgt_emb
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def build_translator(self, metric, method, **kwargs):
        self.dist_function = lambda X, Y: distance_function(X, Y, metric)
        self.method = method
        batch = kwargs.get('batch', None)
        self.batch = batch
        epsilon = kwargs.get('epsilon', None)
        k = kwargs.get('k', None)
        lr = kwargs.get('lr', None)
        iters = kwargs.get('iters', None)

        if self.method == 'nn':
            self.normalizer = None
            self.score_function = lambda q: -self.dist_function(q,
                                                self.tgt_space)

        elif self.method == 'isf':
            self.normalizer = []
            for j in range(0, self.n, batch):
                self.normalizer.append(
                    logsumexp(-self.dist_function(self.src_space,
                              self.tgt_space[j:min(j+batch, self.n)])/epsilon,
                              axis=0))
            self.normalizer = xp.hstack(self.normalizer)
            self.score_function = lambda q: -self.dist_function(q,
                                                self.tgt_space)/epsilon\
                                            - self.normalizer

        elif self.method == 'csls':
            self.normalizer = []
            for j in range(0, self.n, batch):
                dist = self.dist_function(
                            self.src_space,
                            self.tgt_space[j:min(j+batch, self.n)]
                       )
                neighbors, neighborhood = top_k(dist, k, axis=0, biggest=False)
                self.normalizer.append(xp.mean(neighborhood, axis=0))
            self.normalizer = xp.hstack(self.normalizer)
            self.score_function = lambda q: 0.5 * self.normalizer\
                                    - self.dist_function(q, self.tgt_space)

        elif self.method == 'hnn':
            hnn = HNN(self.src_space, self.tgt_space, self.dist_function,
                      epsilon)
            hnn.gallery_weight(iters=iters, batch=batch, lr=lr)
            self.normalizer = hnn.beta
            self.score_function = lambda q: self.normalizer - \
                                        self.dist_function(q, self.tgt_space)

    def translate(self, report_on, dump_all_translations=None):
        """
            Report top-1, 5 and 10 accuracies on the `report_on` set,
            also dump top-10 translations of the entire src vocab if required

            -Input:
            report_on: a dictionary of queries and the corresponding
                       translations, all are word ids. The queries are
                       a subset of the source vocab
            dump_all_translations: If a file path, dump into it the top-10 
                       translations for all the words in source vocab.
                       Otherwise, just report accuracies on the `report_on` set

            -Output:
            top-1, 5, and 10 accuracies
        """
        nq = len(report_on)
        hits = 0

        if dump_all_translations is None:
            # just report on queries (a subset of the src vocab)
            query, truth = zip(*report_on.items())
            query, truth = list(query), list(truth)
            for i in range(0, nq, self.batch):
                this_slice = slice(i, min(i+self.batch, nq))
                scores = self.score_function(self.src_space[query[this_slice]])
                this_retrieval, _ = top_k(scores, 10, axis=1, biggest=True)
                this_acc = compute_precision(this_retrieval, truth[this_slice])
                hits += this_acc * (min(i+self.batch, nq) - i)
            
        else:
            # translate for entire src vocab, report accuracy on the
            # `report_on` set
            create_path_for_file(dump_all_translations)
            tr_file = open(dump_all_translations, 'w')
            for i in range(0, self.n, self.batch):
                this_slice = slice(i, min(i+self.batch, self.n))
                scores = self.score_function(self.src_space[this_slice])
                this_retrieval, _ = top_k(scores, 10, axis=1, biggest=True)
                self.dump_translation(
                        tr_file, slice_to_list(this_slice), this_retrieval)

                in_query = set(report_on.keys()) & \
                        set(slice_to_list(this_slice))
                if in_query:
                    in_query = list(in_query)
                    in_query_mod_batch = [l % self.batch for l in in_query]
                    this_acc = compute_precision(
                                this_retrieval[in_query_mod_batch],
                                [report_on[ii] for ii in in_query])
                    hits += this_acc * len(in_query)
            tr_file.close()
        return hits / nq
       
    def dump_translation(self, fid, src_ids, retrieval_ids):
        assert len(src_ids) == len(retrieval_ids)
        for i, src_id in enumerate(src_ids):
            src_w = self.src_vocab.query_word(src_id)
            tgt_w = [self.tgt_vocab.query_word(j) for j in \
                     retrieval_ids[i].tolist()]
            fid.write(src_w + ' ' + ' '.join(tgt_w) + '\n')
