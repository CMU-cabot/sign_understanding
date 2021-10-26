# Copyright (c) 2021  IBM Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import sklearn
from sklearn import cluster
# from sklearn.cluster.spectral import spectral_clustering



# compute laplacian matrix
def compute_laplacian(A, type_="unnormalized", lambda_=0.):
    n = A.shape[0]
    A = A - np.diag(np.diag(A))
    d = np.sum(A, axis=0)
    
    # compute unnormalized laplacian
    L = np.diag(d) - A
    
    if type_ == "normalized":   
        
        # lambda as a kind of regularizer
        d = (1-lambda_)*d+lambda_
        
        # compute diag of D
        with np.errstate(divide='ignore', invalid='ignore'):
            d_ = 1/(np.sqrt(d)+1e-8)
        d_[np.logical_not(np.isfinite(d_))] = 0.
        
        # compute laplacian matrix
        L = ((np.diag(d_) @ L @ np.diag(d_)))
        
    return L



# estimate number of clusters from normalized Laplacian matrix
def estimate_number_clusters(L_norm, thres=0.5):
    s = np.linalg.svd(L_norm, compute_uv=False)
    n_clusters = np.sum(s < thres)
    return n_clusters




def cluster_affinity(A, thres_cluster=0.5, lambda_=0., debug=False, n_clusters=None):
    """
    Cluster affinity matrix where each element is in [0,1], 1 means 
    highly likely they are in the same clusters.
    The number of clusters is automatically estimated from the number
    of low eigenvalues of the normalized Laplacian matrix.
    Note that diagonal elements are not used

    A:                  input affinity matrix
    thres_cluster:      threshold for cutting off singular values
    thres_single:       threshold for deciding if a point has no neighbor
    lambda_:            set regularizer for normalized laplacian computation
    n_clusters:         number of clusters, will be estimated if None is given
    """
    
    if A.shape[0] == 0:
        return []
    
    # symmetrize
    A = (A + A.T) / 2
    A = A-np.diag(np.diag(A))
    
    # estimate number of clusters
    L_norm = compute_laplacian(A, type_="normalized", lambda_=lambda_)
    if n_clusters is None:
        n_clusters = estimate_number_clusters(L_norm, thres=thres_cluster)
    
    # spectral clustering
    u, s, v = np.linalg.svd(L_norm)
    u_tmp = u[:,-n_clusters:]
    u_tmp = u_tmp/np.linalg.norm(u_tmp, axis=1, keepdims=True)
    labels_ = cluster.KMeans(n_clusters=n_clusters).fit(u_tmp).labels_
    
    if debug:
        print("n_clusters: {}".format(n_clusters))
        return labels_, L_norm
    else:
        return labels_
