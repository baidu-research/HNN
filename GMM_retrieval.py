import argparse
import pylab
from scipy import stats
from HNN import HNN
from function_utils import softmax, logsumexp
try:
    import cupy as xp
    gpu = True
    import numpy as np
except ImportError:
    import numpy as xp
    gpu = False
    np = xp

np.random.seed(7)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=10000,
                        help='number of classes')
    parser.add_argument('-d', type=int, default=300,
                        help='dimension of data')
    parser.add_argument('-q', type=int, default=10000,
                        help='number of queries, evenly distributed over all '
                        'classes')
    parser.add_argument('-g', type=int, default=10000,
                        help='number of gallery examples, evenly distributed '
                        'over all classes')
    parser.add_argument('--sigma', type=float, default=1e-2,
                        help='variance of Gaussian')
    parser.add_argument('--iters', type=int, default=100,
                        help='number of iterations in HNN solver')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='entropy regularizer')
    parser.add_argument('--majority_vote', action='store_true',
                        help='If not given, retrieval is considered as '
                        'correct as long as top k include the right index')
    parser.add_argument('--save_matrices', help='path to save the assigment '
                        'matrices, and the hubness penalizers.')

    args = parser.parse_args()
    print("Gaussian Mixture: c={}, variance={}".format(args.c, args.sigma),
          flush=True)
    print("Data dimension: {}".format(args.d), flush=True)
    print("Queries per class: {}".format(args.q//args.c), flush=True)
    print("Gallery examples per class: {}".format(args.g//args.c), flush=True)
    return args

def gen_data(C, d, Q, G, sigma):
    """ data generation is in CPU. If GPU is available, move data there """
    X_query = []
    X_gallery = []
    n_query_per_class = Q // C
    n_gallery_per_class = G // C

    for c in range(C):
        mu = np.random.uniform(-1, 1, d)
        mu /= np.sqrt(np.sum(mu**2))
        X_query.append(np.random.multivariate_normal(
                       mu, sigma * np.eye(d), n_query_per_class))
        X_gallery.append(np.random.multivariate_normal(
                         mu, sigma * np.eye(d), n_gallery_per_class))

    X_query = np.vstack(X_query)
    X_gallery = np.vstack(X_gallery)
    if xp != np:
        X_query = xp.array(X_query)
        X_gallery = xp.array(X_gallery)
    y_query = np.hstack([[i] * n_query_per_class for i in range(C)])
    y_gallery = np.hstack([[i] * n_gallery_per_class for i in range(C)])

    if gpu:
        X_query = xp.asarray(X_query)
        X_gallery = xp.asarray(X_gallery)

    return X_query, X_gallery, y_query, y_gallery

def NN_softmax_form(D, epsilon, axis=1):
    """
        Return P_{i,j} \propto exp(-D_{i,j}/epsilon)
        axis=0, normalize column-wise
        axis=1, normalize row-wise
    """
    W = -D / epsilon
    return softmax(W, axis=axis)

def isoftmax(D, epsilon, axis=1):
    """
        Given a 2D distance matrix D, compute inverted softmax.
        Along the axis must add to 1
    """
    m, n = D.shape
    P = NN_softmax_form(D, epsilon, axis=1-axis)
    P /= xp.sum(P, axis=axis, keepdims=True)
    hub_penalizer = logsumexp(-D/epsilon, axis=1-axis)
    return P, hub_penalizer

def HNN_primal(D, epsilon, iters=100, compute_accuracy=None):
    """
        Sinkhorn Solver for the following problem of mxn matrix P:
        min_P <D, P> + epsilon * H(P)
        s.t.  P >= 0, \sum_j P_{i,j} =1, \sum_i P_{i,j} = m/n
    """
    m, n = D.shape
    P = NN_softmax_form(D, epsilon, axis=0)
    P /= xp.sum(P, axis=1, keepdims=True)
    if compute_accuracy is not None:
        acc = []
        acc.append(compute_accuracy(P))
    for i in range(1, iters):
        P /= xp.sum(P, axis=0, keepdims=True)
        P /= xp.sum(P, axis=1, keepdims=True)
        if compute_accuracy is not None:
            acc.append(compute_accuracy(P))
    if compute_accuracy is None:
        return P
    else:
        return P, acc

def compute_accuracy(P, g_labels, q_labels, majority=True):
    """
        Takes in assignment matrix P, and return the classfication accuracy
        The calculation is done on CPU
    """
    p = []
    if xp != np:
        P = xp.asnumpy(P)
    for k in [1, 5, 10]:
        hits = 0
        predicts = np.argpartition(-P, k, axis=1)[:, :k]
        if not majority:
            # as long as the correct class is included
            for i in range(predicts.shape[0]):
                predicts[i] = g_labels[predicts[i]]
                if q_labels[i] in predicts[i].tolist():
                    hits += 1
            p.append(hits / predicts.shape[0] * 100)
        else:
            for i in range(predicts.shape[0]):
                predicts[i] = g_labels[predicts[i]]
            p_label, _ = stats.mode(predicts, axis=1)
            p.append(np.mean(p_label.flatten() == np.array(q_labels)) * 100)
    return p

if __name__ == '__main__':
    args = parse_args()
    print("Generating data ...", flush=True)
    X_query, X_gallery, y_query, y_gallery = gen_data(args.c, args.d,
                                                      args.q, args.g,
                                                      args.sigma)
    print("Done data generation", flush=True)
    def dist_function(a, b):
        assert len(a.shape) == 2
        assert len(b.shape) == 2
        a_norms = xp.sum(a**2, axis=1, keepdims=1)
        b_norms = xp.sum(b**2, axis=1, keepdims=1)
        return a_norms + b_norms.T - 2 * a.dot(b.T)
    dist_mtx = dist_function(X_query, X_gallery)

    # NN
    P_NN = NN_softmax_form(dist_mtx, args.epsilon, axis=1)
    p_nn = compute_accuracy(P_NN, y_gallery, y_query, args.majority_vote)
    print("NN test accuracy (top-[1, 5, 10]): {}%".format(p_nn), flush=True)

    # ISF
    P_ISF, hp_isf = isoftmax(dist_mtx, args.epsilon, axis=1)
    p_isf = compute_accuracy(P_ISF, y_gallery, y_query, args.majority_vote)
    print('ISF test accuracy (top-[1, 5, 10]): {}%'.format(p_isf), flush=True)

    # HNN primal
    P_HNN0 = HNN_primal(dist_mtx, args.epsilon, args.iters)
    p_hnn0 = compute_accuracy(P_HNN0, y_gallery, y_query, args.majority_vote)
    print('HNN primal test accuracy (top-[1, 5, 10]): {}%'.format(p_hnn0),
          flush=True)

    # HNN dual
    print("Running HNN dual ...", flush=True)
    HNN_dual = HNN(X_query, X_gallery, dist_function, args.epsilon)
    HNN_dual.gallery_weight(args.iters, batch=128, lr=100)
    HNN_dual.get_full_plan()
    hp_hnn = -HNN_dual.beta / args.epsilon
    P_HNN1 = HNN_dual.P
    p_hnn1 = compute_accuracy(P_HNN1, y_gallery, y_query, args.majority_vote)
    print('HNN dual test accuracy (top-[1, 5, 10]): {}%'.format(p_hnn1),
          flush=True)

    if args.save_matrices:
        xp.savez(args.save_matrices, P_NN=P_NN, P_ISF=P_ISF,
                 P_HNN_primal=P_HNN0, P_HNN_dual=P_HNN1,
                 hp_isf=hp_isf, hp_hnn=hp_hnn)
