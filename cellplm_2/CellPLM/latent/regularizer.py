import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sp

def torch_corr(x, dim=0):
    if dim == 1:
        mean_x = torch.mean(x, 1)
        xm = x - mean_x.view(-1, 1)
        c = xm.mm(xm.t())

    elif dim == 0:
        mean_x = torch.mean(x, 0)
        xm = x - mean_x.view(1, -1)
        c = xm.t().mm(xm)
    else:
        raise ValueError('dim must be 0 or 1')

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    c = c.mean()
    return c

def get_pairwise_sim(x):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass

    if sp.issparse(x):
        x = x.todense()
        x = x / (np.sqrt(np.square(x).sum(1))).reshape(-1,1)
        x = sp.csr_matrix(x)
    else:
        x = x / (np.sqrt(np.square(x).sum(1))+1e-10).reshape(-1,1)
    # x = x / x.sum(1).reshape(-1,1)
    try:
        dis = euclidean_distances(x)
        return 0.5 * (dis.sum(1)/(dis.shape[1]-1)).mean()
    except:
        return -1