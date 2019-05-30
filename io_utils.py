import os
from functools import reduce
from gensim.models import KeyedVectors
try:
    import cupy as xp
    gpu = True
except ImportError:
    import numpy as xp
    gpu = False

class vocab:
    def __init__(self, words):
        self.word2id = dict([(w, i) for i, w in enumerate(words)])
        self.id2word = dict([(i, w) for i, w in enumerate(words)])

    def __len__(self):
        return len(self.word2id)

    def query_id(self, word):
        """ Return the index of a word in the vocab
            If the word is not in the vocab, return None.
        """
        return self.word2id.get(word, None)

    def query_word(self, ind):
        """ Return the word given the index """
        return self.id2word.get(ind, "<UNKOWN-WORD>")


def load_embedding(emb_file, max_size=None):
    word_vecs = KeyedVectors.load_word2vec_format(emb_file, binary=False,
                                                  limit=max_size)
    embeddings = xp.array(word_vecs.vectors)
    return embeddings, vocab(word_vecs.index2word)


def load_seeding_dict(dic_file, src_vocab, tgt_vocab):
    """Load seeding dictionary, return two lists of the same length, 
       src_w[i] translates to tgt_w[i]. 
       
       If the source/target word is not in the src_vocab,
       then the pair is excluded.
    """
    pairs = []
    with open(dic_file, 'rb') as f:
        for line in f:
            try:
                this_line = line.decode('utf-8')
            except:
                this_line = line.decode('latin-1')
            src_w, tgt_w = this_line.rstrip().split()
            src_w_id = src_vocab.query_id(src_w)
            tgt_w_id = tgt_vocab.query_id(tgt_w)
            if src_w_id is not None and tgt_w_id is not None:
                pairs.append((src_w_id, tgt_w_id))
    
    src_w, tgt_w = zip(*pairs)
    print("Found {} pairs of words in seeding dictionary: "
          "{} unique src words; {} unique tgt words.".format(
          len(pairs), len(set(src_w)), len(set(tgt_w))), flush=True)
    return list(src_w), list(tgt_w)


def load_queries(dic_file, src_vocab, tgt_vocab,
                 training_src_words=None, max_query=None):
    """Load queries and their ``groundtruth`` translations for testing.
       Exclude the pair if the query appears in training, or if the 
       query/translation is out of vocab

       Return:
       queries: a dictionary where
            queries[src_w_id] = [tgt_w_id1, tgt_w_id2, ...],
       where src_w_id can have multiple target translations, stored in the list
    """
    queries = {}
    cnt = 0
    with open(dic_file, 'rb') as f:
        for line in f:
            try:
                this_line = line.decode('utf-8')
            except:
                this_line = line.decode('latin-1')
            src_w, tgt_w = this_line.rstrip().split()
            src_w_id = src_vocab.query_id(src_w)
            tgt_w_id = tgt_vocab.query_id(tgt_w)
            if src_w_id is not None and tgt_w_id is not None:
                if training_src_words is None or \
                        src_w_id not in training_src_words:
                    if src_w_id in queries:
                        queries[src_w_id].append(tgt_w_id)
                    else:
                        cnt += 1
                        if max_query is None or cnt <= max_query:
                            queries[src_w_id] = [tgt_w_id]
    num_pairs = len(reduce(lambda a,b: a+b, queries.values()))
    total_tgt_words = len(set(reduce(lambda a,b: a+b, queries.values())))
    print("{} queries loaded, {} target translations involved. "
          "{} pairs in total.".format(cnt, total_tgt_words, num_pairs),
          flush=True)
    return queries


def create_path_for_file(filename):
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)
