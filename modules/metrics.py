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
from typing import List
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix
import editdistance
from scipy.stats import kendalltau

def compute_avg_metric_tpfp(list_gt_mat: List[np.array], 
                            list_pred_mat: List[np.array], 
                            thres: np.array):
    """
    Compute TP/FP/FN/TN from the upper triangular entries of the 
    input matrices.
    
    INPUT
    --------
    list_gt_mat    : n x n np.array    - list of ground truth matrices (will be thesholded at 0.5)
    list_pred_mat  : n x n np.array    - list of predicted matrices (values between 0 and 1)
    thres          : m np.array        - list of threshold for thresholding pred_vec
    
    OUTPUT
    --------
    conf_mat       : 4 x m np.array    - TP/FP/FN/TN as a ratio w.r.t number of entries for each threshold value
    
    EXAMPLE
    -------
    list_gt_mat = []
    list_pred_mat = []
    thres = np.arange(0.1, 1, 0.1)

    for i in range(20):
        ii = np.random.randint(10, 20)
        list_gt_mat.append(np.random.rand(ii, ii)) # not a symmetric matrix
        list_pred_mat.append(np.random.rand(ii, ii)) # not a symmetric matrix

    conf_mat = compute_avg_metric_tpfp(gt_mat, pred_mat, thres)
    """

    conf_mat = np.zeros((4, len(thres), len(list_gt_mat)))
    
    for i in range(len(list_gt_mat)):
        conf_mat[:,:,i] = compute_metric_tpfp(list_gt_mat[i], list_pred_mat[i], thres)
        
    conf_mat = np.mean(conf_mat, axis=2)
    return conf_mat


def compute_metric_tpfp(gt_mat: np.array, 
                        pred_mat: np.array, 
                        thres: np.array):
    """
    Compute TP/FP/FN/TN from the upper triangular entries of the 
    input matrices.
    
    INPUT
    --------
    gt_mat    : n x n np.array    - ground truth matrix (will be thesholded at 0.5)
    pred_mat  : n x n np.array    - predicted matrix (values between 0 and 1)
    thres     : m np.array        - list of threshold for thresholding pred_vec
    
    OUTPUT
    --------
    conf_mat  : 4 x m np.array    - TP/FP/FN/TN as a ratio w.r.t number of entries for each threshold value
    """
    
    assert np.all(np.logical_and(0 <= pred_mat, pred_mat <= 1))
    
    n = pred_mat.shape[0]
    conf_mat   = np.zeros((4, len(thres))) # [[TP, FP, FN, TN]]

    pred_vec   = pred_mat[np.triu_indices(n, 1)]
    gt_vec     = gt_mat[np.triu_indices(n, 1)] > 0.5
    gt_vec_neg = np.logical_not(gt_vec)

    for i, th in enumerate(thres):
        pred_vec_thres     = pred_vec > th
        pred_vec_thres_neg = np.logical_not(pred_vec_thres)

        conf_mat[0, i]  = np.sum(np.logical_and(gt_vec, pred_vec_thres)) # TP
        conf_mat[1, i]  = np.sum(np.logical_and(gt_vec_neg, pred_vec_thres)) # FP
        conf_mat[2, i]  = np.sum(np.logical_and(gt_vec, pred_vec_thres_neg)) # FN
        conf_mat[3, i]  = np.sum(np.logical_and(gt_vec_neg, pred_vec_thres_neg)) # TN
        
    conf_mat = conf_mat / len(gt_vec)
    
    return conf_mat


def compute_cluster_accuracy(gt_group_id, pred_group_id):
    """
    Compute clustering accuracy. Works with arbitrary number
    of clusters for both inputs, and also invariant to 
    permutation of cluster numbers.
    
    INPUT
    ------
    gt_group_id    : n np.array     - groud truth group id of each member
    pred_group_id  : n np.array     - predicted group id of each member
    
    OUTPUT
    ------
    clus_acc       : float          - clustering accuracty in [0, 1] (higher is better)
    
    EXAMPLE
    ------
    gt_group_id = np.random.randint(0, 5, size=20)
    pred_group_id = np.random.randint(0, 15, size=20)
    clus_acc = compute_cluster_accuracy(gt_group_id, pred_group_id)
    """
    
    assert len(gt_group_id) == len(pred_group_id)
    
    # count 
    gt_count = np.unique(gt_group_id, return_counts=True)[1]
    pred_count = np.unique(pred_group_id, return_counts=True)[1]

    # coorccurence matrix
    coor_mat = np.minimum(gt_count[None,:], pred_count[:,None])

    # assignment (Hungarian)
    max_idx = linear_sum_assignment(coor_mat, maximize=True)

    # compute the clustering accuracy
    clus_acc = np.sum(coor_mat[max_idx])/np.sum(gt_count)

    return clus_acc


def node_class_accuracy(y_true, y_pred):
    """
    Compute node class accuracy using accuracy_score from sklearn.metrics
    
    INPUT
    ------
    y_true  : np.array     - ground truth class
    y_pred  : np.array     - predicted class
    
    OUTPUT
    ------
    acc     : float        - accuracy in [0,1]
    
    """
    
    return accuracy_score(y_true, y_pred)


def node_class_confmat(y_true, y_pred, labels_list):
    """
    Compute node class accuracy using accuracy_score from sklearn.metrics
    
    INPUT
    ------
    y_true  : np.array     - ground truth class
    y_pred  : np.array     - predicted class
    labels_list : np.array - list of class labels
    
    OUTPUT
    ------
    acc     : float        - accuracy in [0,1]
    
    """
    
    return confusion_matrix(y_true, y_pred, labels=labels_list)


def compute_pair_accuracy(gt_order, pred_order, circ=False):
    """
    Compute accuracy of a predicted ordering by finding the 
    ratio of the correct pairs.
    I.e.   acc  = (np. of correct pairs)/(no. of all pairs )
    
    INPUT
    ------
    gt_order    : n np.array   - input ground truth list
    pred_order  : n np.array   - predicted list
    circ        : bool         - consider cicular (cycle) or path
    
    OUTPUT
    ------
    acc         : float        - accuracy of the predicted ordering
    """

    assert len(gt_order) == len(pred_order)
    
    # get list of all gt pairs
    n_gt = len(gt_order)
    n_edges = len(gt_order)-1+circ
    gt_pairs = {(gt_order[i],gt_order[(i+1) % n_gt]) for i in range(n_edges)}
    
    # count number of pairs in the gt_pair
    count = 0
    for i in range(n_edges):
        p1, p2 = pred_order[i], pred_order[(i+1) % n_gt]
        if ((p1, p2) in gt_pairs) or ((p2, p1) in gt_pairs):
            count += 1
        
        
    return count / n_edges


