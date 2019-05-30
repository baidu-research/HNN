import argparse
from io_utils import *
from scorer import scorer
try:
    import cupy as xp
    gpu = True
except ImportError:
    import numpy as xp
    gpu = False

def parse_args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawTextHelpFormatter)
    # data loading
    data_args = parser.add_argument_group('Data Options')
    data_args.add_argument('src_emb', type=str,
                           help='path to src embedding file')
    data_args.add_argument('tgt_emb', type=str,
                           help='path to tgt embedding file')
    data_args.add_argument('-m', '--src_vocab_size', type=int, default=200000,
                           help='source vocab size')
    data_args.add_argument('-n', '--tgt_vocab_size', type=int, default=200000,
                           help='target vocab size')
    data_args.add_argument('-d', '--seed_dict', type=str,
                           help='path to training dictionary file')
    data_args.add_argument('-t', '--test_dict', type=str,
                           help='path to test dictionary file, the overlap '
                           'with training dict will be removed when '
                           'calculating translation accuracy')
    data_args.add_argument('--max_query', type=int,
                           help='maximum number of queries if given')

    # Linear Mapping Options
    mapping_args = parser.add_argument_group('Linear Mapping Options')
    mapping_args.add_argument('--mapping', type=str,
                              choices=['l2', 'hinge'], default='l2',
                              help='Choice of Linear Mapping Learner:\n'
                              'l2: min \sum_i |W * x_i - y_i|^2 \n'
                              'hinge: min \sum_{i,j} '
                              'max{0, 1 + <W * x_i, y_j> - <W * x_i, y_i>}')
    mapping_args.add_argument('--orth', action='store_true',
                              help='whether to contrain the mapping to be '
                              'orthonormal.')

    # Retrieval Options
    retrieval_args = parser.add_argument_group('Retrieval Options')
    retrieval_args.add_argument('--metric', type=str,
                                choices=['cos', 'sqeuc'], default='cos',
                                help='choice of distance metric for retrieval.'
                                ' Default to cosine distance')
    retrieval_args.add_argument('--method', type=str,
                                choices=['nn', 'csls', 'isf', 'hnn'],
                                default='nn',
                                help='Choice of retrieval method.\n'
                                'nn: Vanilla nearest neighbor\n'
                                'csls: Cross-domain Local Scaling\n'
                                'isf: inverted softmax\n'
                                'hnn: Hubless Nearest Neighbor\n'
                                'Default to nn.')
    retrieval_args.add_argument('--batch', type=int, default=128,
                                help='computing distances in batch of '
                                'queries or gallery examples, to avoid '
                                'memory overflow')
    retrieval_args.add_argument('--knn', type=int, default=10,
                                help='number of nearest neighbors to estimate '
                                'hubness, parameter for csls only. '
                                'Default to 10')
    retrieval_args.add_argument('--epsilon', type=float, default=1./30,
                                help='heat kernel parameter for '
                                'inverted softmax and HNN. Default to 1/30')
    retrieval_args.add_argument('--iters', type=int, default=30,
                                help='number of batch gradient steps in '
                                'HNN solver. Default: 30')
    retrieval_args.add_argument('--lr', type=float, default=1e4,
                                help='learning rate for gradient steps in HNN')
    
    # Logging and Checkpoint Options
    ckpt_args = parser.add_argument_group('Logging and Checkpoint Options')
    ckpt_args.add_argument('--save_translation', type=str,
                           help='path to save top 10 '
                           'translations for every source word')

    args = parser.parse_args()

    # print setups
    _, f = os.path.split(args.seed_dict)
    task = f.split('.')[0]
    print("Task: {}".format(task), flush=True)
    print("GPU: {}".format(gpu), flush=True)
    print("Src_emb_file: {}".format(args.src_emb), flush=True)
    print("Tgt_emb_file: {}".format(args.tgt_emb), flush=True)
    print("Max_src_vocab_size: {}".format(args.src_vocab_size), flush=True)
    print("Max_tgt_vocab_size: {}".format(args.tgt_vocab_size), flush=True)
    print("Seeding dictionary: {}".format(args.seed_dict), flush=True)
    print("Queries and Ground-truths: {}".format(args.test_dict), flush=True)
    print("Upper limit on the number of query items: {}".format(
          args.max_query), flush=True)
    print("Procrustes: {}".format(args.mapping=='l2' and args.orth),
          flush=True)
    print("Retrieval metric: {}".format(args.metric), flush=True)
    print("Retrieval method: {}".format(args.method), flush=True)
    if args.method == 'isf':
        print("Entropy regularizer: {}".format(args.epsilon), flush=True)
    elif args.method == 'hnn':
        print("Entropy regularizer: {}".format(args.epsilon), flush=True)
        print("Learning rate: {}".format(args.lr), flush=True)
        print("Number of iterations {}".format(args.iters), flush=True)
    if args.save_translation is not None:
        print("Save top-10 translations to {}".format(args.save_translation),
              flush=True)
    print()
    return args

def learning_mapping(src_vec, tgt_vec, method='l2', orth=True):
    if method == 'l2':
        if orth:
            # procrustes problem
            U, _, Vh = xp.linalg.svd(src_vec.T.dot(tgt_vec))
            W = U.dot(Vh)
        else:
            # least squares
            W = xp.linalg.lstsq(src_vec, tgt_vec, rcond=None)[0]
    elif method == 'hinge':
        raise NotImplementedError
    return W

def main():
    args = parse_args()

    print("Loading source embeddings and building source vocab ...",
          flush=True)
    src_emb, src_vocab = load_embedding(args.src_emb, args.src_vocab_size)
    print("Source vocab size: {}".format(len(src_vocab)), flush=True)

    print("Loading target embeddings and building target vocab ...",
          flush=True)
    tgt_emb, tgt_vocab = load_embedding(args.tgt_emb, args.tgt_vocab_size)
    print("Target vocab size: {}".format(len(tgt_vocab)), flush=True)

    print("Loading seeding dictionary ...", flush=True)
    train_src_words, train_tgt_words = load_seeding_dict(args.seed_dict,
                                                         src_vocab, tgt_vocab) 

    print("Loading queries ...", flush=True)
    queries = load_queries(args.test_dict, src_vocab, tgt_vocab,
                           train_src_words, args.max_query)

    print("Learning Linear Mapping ...", flush=True)
    W = learning_mapping(src_emb[train_src_words],
                         tgt_emb[train_tgt_words],
                         args.mapping, args.orth)

    print("Building {} scorer ...".format(args.method.upper()), flush=True)
    S = scorer(W, src_emb, tgt_emb, src_vocab, tgt_vocab)
    S.build_translator(args.metric, args.method,
                       k=args.knn, epsilon=args.epsilon, batch=args.batch,
                       iters=args.iters, lr=args.lr)

    print("Querying ...", flush=True)
    test_precision = S.translate(queries, args.save_translation)
    print()
    print("----Retrieval accuracy ----", flush=True)
    print("P@1(%): {}, P@5(%): {}, P@10(%): {}".format(
          test_precision[0], test_precision[1], test_precision[2]),
          flush=True)
    print("Done")

if __name__ == '__main__':
    main()
