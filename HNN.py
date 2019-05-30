import time
from function_utils import softmax, logsumexp
try:
    import cupy as xp
    gpu = True
except ImportError:
    import numpy as xp
    gpu = False

class HNN():
    def __init__(self, src, tgt, dist_function, epsilon):
        self.src = src  # m x d
        self.tgt = tgt  # n x d
        self.m = src.shape[0]
        self.n = tgt.shape[0]
        assert src.shape[1] == tgt.shape[1]
        self.dist_function = dist_function
        self.epsilon = epsilon

    def gallery_weight(self, iters=1000, batch=32, lr=1e3, ini_beta=None):
        """ Get the weights for all gallery items
        """
        if ini_beta is None:
            beta = xp.zeros(self.n)
        else:
            beta = ini_beta 

        def grad_over_batch(sample, beta):
            r = (beta - self.dist_function(
                        self.src[sample], self.tgt)) / self.epsilon
            probs = softmax(r, axis=1)
            grad = 1. / self.n - xp.mean(probs, axis=0)
            return grad

        def grad_over_all(beta):
            G = xp.zeros(self.n)
            for i in range(0, self.m, batch):
                sample_ids = xp.arange(i, min(i+batch, self.m))
                G += grad_over_batch(sample_ids, beta) * len(sample_ids)
            return G / self.m

        """
        # Re-implemented pyOT, batchsize == 1
        cur_beta = xp.zeros(self.n)
        ave_beta = xp.zeros(self.n) # the column-wise normalizer in log domain
        for i in range(iters):
            k = i + 1
            sample_i = xp.random.randint(self.m)
            r = (cur_beta - self.dist_function(
                        self.src[sample_i], self.tgt)) / self.epsilon
            probs = softmax(r)
            grad = 1. / self.n - probs
            cur_beta += (lr / xp.sqrt(k)) * grad
            ave_beta = (1. / k) * cur_beta + (1 - 1. / k) * ave_beta
            loss_i = self.epsilon * logsumexp(r) - xp.mean(ave_beta)
            loss.append(loss_i)

        self.beta = ave_beta
        """
        for i in range(iters):
            t0 = time.time()
            g = grad_over_all(beta)
            beta += lr * g
            delta_t = time.time() - t0
            gnorm_i = xp.linalg.norm(g)
            print("Iter: {}, grad norm: {}, time: {}".format(
                  i+1, gnorm_i, delta_t), flush=True)
        self.beta = beta

    def get_full_plan(self):
        # c-transform
        M = self.dist_function(self.src, self.tgt)
        self.alpha = -self.epsilon * \
                        logsumexp((self.beta[None, :] - M) / self.epsilon, 1) \
                     + self.epsilon * xp.log(self.n)
        self.P = xp.exp((self.alpha[:, None] +
                         self.beta[None, :] - M) / self.epsilon) / self.n

    def get_scores_for_query(self, query_id):
        M = self.dist_function(self.src[query_id], self.tgt)
        scores = self.beta - M
        return scores
