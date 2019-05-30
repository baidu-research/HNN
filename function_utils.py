try:
    import cupy as xp
    gpu = True
except ImportError:
    import numpy as xp
    gpu = False
    from scipy.misc import logsumexp as scp_logsumexp

def slice_to_list(s):
    """ map a slice object to list """
    return [i for i in range(s.start, s.stop, s.step if s.step else 1)]

def softmax(W, axis=0):
    """ Return w_i <- e^w_i / sum_j e^w_j
        axis=0, normalize every column
        axis=1, normalize every row
    """
    W = W - W.max(axis=axis, keepdims=True)
    W = xp.exp(W)
    return W / xp.sum(W, axis=axis, keepdims=True)

def logsumexp(W, axis=0, keepdims=False):
    if gpu:
        max_along_axis = W.max(axis=axis, keepdims=1)
        v = max_along_axis + \
              xp.log(xp.sum(xp.exp(W - max_along_axis), axis=axis, keepdims=1)) 
        if not keepdims:
            v = xp.squeeze(v)
        return v
    else:
        return scp_logsumexp(W, axis=axis, keepdims=keepdims)

def distance_function(X, Y, metric='cos', output_as_2d=True):
    """
    Re-implementing scipy.spatial.distance.cdist, as cupy does not have
    this interface.

    Also, as an extension of the cdist, this implementation allows 1-d array(s)
    as input(s). If one of X and Y is 1-d, then output is 1-d. If both are 1-d, 
    then output is a scalar.

    If output_as_2d is True, force 2d output
    """
    # first ensure dimensions are 2d
    if X.ndim == 1:
        X = X[xp.newaxis, :]
    if Y.ndim == 1:
        Y = Y[xp.newaxis, :]
    assert X.ndim == 2 and Y.ndim == 2
    assert X.shape[1] == Y.shape[1]

    X_norm_squared = xp.sum(X**2, axis=1, keepdims=1)
    Y_norm_squared = xp.sum(Y**2, axis=1, keepdims=1)
    X_dot_Y = X.dot(Y.T)
    if metric == 'sqeuc':
        D = X_norm_squared + Y_norm_squared.T - 2 * X_dot_Y
    elif metric == 'cos':
        D = 1 - X_dot_Y / xp.sqrt(X_norm_squared) / xp.sqrt(Y_norm_squared.T)

    # If not forced 2d output, squeeze dimension
    if not output_as_2d:
        D = D.squeeze()
    return D

def top_k(array, k, axis=0, biggest=True):
    """ Return the topK index along the specified dimension,
        The returned indices are such that their array values are sorted
        
        -Input:
        array: 1d or 2d array
        k: the top `k` (k>0, integer)
        axis: futile if array is 1d, otherwise sorting along the specified axis
              default to 0
        biggest: whether the top-k biggest or smallest, default to True

        -Output:
        inds: indices
        vals: array values at the indices
    """
    assert array.ndim == 1 or array.ndim == 2
    assert axis == 0 or axis == 1
    if biggest:
        array = -array
    
    if array.ndim == 1:
        inds = xp.argpartition(array, k)[:k]
        vals = array[inds]
        sort_inds = xp.argsort(vals)
        inds = inds[sort_inds]
        vals = vals[sort_inds]

    elif axis == 0:
        inds = xp.argpartition(array, k, axis=0)[:k, :]
        vals = array[inds, xp.arange(array.shape[1])[None, :]]
        sort_inds = xp.argsort(vals, axis=0)
        inds = inds[sort_inds, xp.arange(array.shape[1])[None, :]]
        vals = vals[sort_inds, xp.arange(array.shape[1])[None, :]]

    else:
        inds = xp.argpartition(array, k, axis=1)[:, :k]
        vals = array[xp.arange(array.shape[0])[:, None], inds]
        sort_inds = xp.argsort(vals, axis=1)
        inds = inds[xp.arange(array.shape[0])[:, None], sort_inds]
        vals = vals[xp.arange(array.shape[0])[:, None], sort_inds]

    if biggest:
        vals = -vals
    return inds, vals

def compute_precision(id_mtx, truth):
    """ Evaluate the `translation` precision-at-k based on retrieved indices
        Refer to 
        https://github.com/facebookresearch/MUSE/blob/master/src/evaluation/word_translation.py
        line 140-149
        Args:
            id_mtx: m by n, i-th row are n retrievals for the i-th query
            truth: an m-list of lists, where truth[i] are translations for
                the i-th source word (corresponding to the i-th row in P)
        return:
            top1, 5 and 10 accuracies
    """
    n_query = len(truth)
    assert id_mtx.shape[0] == n_query
    p_at_k = []
    for k in [1, 5, 10]:
        topk = id_mtx[:, :k]
        hits = 0
        for i, trans in enumerate(truth):
            hits += bool(set(topk[i].tolist()) & set(trans))
        hits /= n_query
        p_at_k.append(hits*100)

    return xp.array(p_at_k) 

def hist_k_occurrence(retrievals, Nbins=40, upperlimit_quantile=1):
    """
        Give the top retrievals, compute the histogram of k-occurrence
        
        -Input:
        retrievals: list of lists, an inner list is retrievals for a query,
                    in preferece decending order
        Nbins: number of bins to calculate the histogram
        upperlimit_quantile: cut off at this quantile of the counts.
                             Default to 1, i.e., max value of the counts

        -Output:
        bins: the bins of the histogram
        freqs: the frequencies at the bins,
               so `plot(bins, freqs)` will plot the k_occurrence's p.d.f
    """
    if gpu:
        import numpy as np
    else:
        np = xp
    if isinstance(retrievals, list):
        retrievals = np.hstack(retrievals)
    else:
        retrievals = retrievals.ravel()
    r, cnt = np.unique(retrievals, return_counts=True)
    if upperlimit_quantile == 1:
        upperlimit = cnt.max()
    else:
        assert upperlimit_quantile < 1 and upperlimit_quantile > 0
        upperlimit = np.quantile(cnt, upperlimit_quantile)
    bins = np.linspace(1, upperlimit, Nbins+1)
    bin_interval = bins[1] - bins[0]
    bin_cnts, bin_edges = np.histogram(cnt, bins)
    freqs = bin_cnts/len(cnt)/bin_interval
    return bin_edges[:-1], freqs
