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

import cv2
# import matplotlib.pyplot as plt
import numpy as np
# from scipy.ndimage import gaussian_filter
import torch
import copy

def softmax(X, axis=1):
    Y = np.exp(X)
    Y = Y/np.sum(Y, axis=axis, keepdims=True)
    return Y


def pw_sq_dist(X, Y):
    # compute pairwise squared distance between all pairs of inputs
    # X : d x n
    # Y : d x m
    # output : n x m
    return np.sum((X.T) ** 2, axis=1, keepdims=True) + \
        np.sum(Y ** 2, axis=0, keepdims=True) + \
        -2*(X.T @ Y)

def tform_bboxes_xyxy2xywh(bboxes):
    return np.stack([
        bboxes[:, 0],
        bboxes[:, 1],
        bboxes[:, 2]-bboxes[:, 0],
        bboxes[:, 3]-bboxes[:, 1],
        ],axis=1)

def tform_bboxes_xywh2xyxy(bboxes):
    return np.stack([
        bboxes[:, 0],
        bboxes[:, 1],
        bboxes[:, 2]+bboxes[:, 0],
        bboxes[:, 3]+bboxes[:, 1],
        ],axis=1)


def scale_bbox_xyxy(bboxes, scale): 
    # assume input bboxes is 4 x n with (min x, min y, max x, max y)
    # scale multiple bboxes with scale
    if len(bboxes) == 0:
        return np.array([])
    
    box_center_x = (bboxes[:, 0] + bboxes[:, 2])/2.0
    box_center_y = (bboxes[:, 1] + bboxes[:, 3])/2.0
    box_rad_x = (bboxes[:, 2] - bboxes[:, 0])/2.0
    box_rad_y = (bboxes[:, 3] - bboxes[:, 1])/2.0
    return np.stack([box_center_x-scale*box_rad_x, \
                           box_center_y-scale*box_rad_y,\
                           box_center_x+scale*box_rad_x,\
                           box_center_y+scale*box_rad_y], axis=1)


def weighted_median(x, weights):
    # solve weighted median and returns the value and index of the weighted median
    # is input x is None, then None is return instead of the median value

    weights_idx = list(zip(weights, np.arange(len(weights))))
    weights_idx = np.array(sorted(weights_idx, key=lambda y: y[0]))
    cs_w = np.cumsum(weights_idx[:, 0]) 
    med_idx = np.argmax(cs_w > cs_w[-1]/2)
    med_idx = weights_idx[med_idx, 1].astype(np.int)
    if x is None:
        return None, med_idx
    else:
        return x[med_idx], med_idx
    

def iou_bboxes(bbox1, bbox2): 
    # Compute IOU between two bboxes  
    # format bboxes : [xtl, ytl, xbr, ybr]
    
    xtl1, ytl1, xbr1, ybr1 = bbox1[:4]
    xtl2, ytl2, xbr2, ybr2 = bbox2[:4]
    
    intersection = min(0, (max(xtl1, xtl2)-min(xbr1, xbr2))) * min(0, (max(ytl1, ytl2)-min(ybr1, ybr2)))
    union = (xbr1-xtl1)*(ybr1-ytl1) + (xbr2-xtl2)*(ybr2-ytl2) - intersection
    return intersection / union


def compute_pairwise_iou(bboxes1, bboxes2):
    # compute pairwise IOU
    # format bboxes : [xtl, ytl, xbr, ybr]
    
    iou = np.zeros((len(bboxes1), len(bboxes2)))
    
    for b1 in range(len(bboxes1)):
        for b2 in range(len(bboxes2)):
            iou[b1, b2] = iou_bboxes(bboxes1[b1], bboxes2[b2])
            assert iou[b1, b2] >= 0, "{} {} {}".format(iou[b1, b2], b1,b2)
            
    return iou

def crop_image_bboxes(img, bboxes):
    # Crop image according to (multiple) given bboxes
    # Assumes input image is RGB float between [0,1]
    img_size = img.shape
    img_cropped = [None]*len(bboxes)
    
    for i, bbox in enumerate(bboxes):
        
        # check size with images
        bbox[0], bbox[2] = np.round(min(max(bbox[0], 0), img_size[1])), np.round(min(max(bbox[2], 0), img_size[1]))
        bbox[1], bbox[3] = np.round(min(max(bbox[1], 0), img_size[0])), np.round(min(max(bbox[3], 0), img_size[0]))
        
        # crop
        bbox = np.array(bbox).astype(np.int)
        img_cropped[i] = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        
    return img_cropped


def suppress_non_max_in_matrix(M, thres=0):
    # input is torch tensor
    # find elements that are the maximum of its row and column
    # then set all other elements to zero
    #
    # returns both suppresses matrix and triplets of [idx_0, idx_1, value] 
    # of the max entries
    #
    # Note, for the triplets, if there are more than 1 pairs with same value
    # in the same row/column, only one entry will be returned (e.g. if all 
    # elements are 0., then only n triplets wil lbe returned instead of n x n)
    
    if isinstance(M, torch.Tensor):
        
        if torch.numel(M) == 0:
            return M, []
        
        val0, idx0 = torch.max(M, dim=0)

        Mout = torch.zeros_like(M, device=M.device)
        Mout[idx0, torch.arange(M.shape[1], dtype=torch.long)] = val0

        val1, idx1 = torch.max(Mout, dim=1)

        Mout = torch.zeros_like(M, device=M.device)
        Mout[torch.arange(M.shape[0], dtype=torch.long), idx1] = val1
        Mout_np = Mout.detach().cpu().numpy()
        
    elif isinstance(M, np.ndarray):
        
        if np.size(M) == 0:
            return M, []
        
        val0 = np.max(M, axis=0)
        idx0 = np.argmax(M, axis=0)
        
        Mout = np.zeros_like(M)
        Mout[idx0, np.arange(M.shape[1])] = val0

        val1 = np.max(Mout, axis=1)
        idx1 = np.argmax(Mout, axis=1)

        Mout = np.zeros_like(M)
        Mout[np.arange(M.shape[0]), idx1] = val1
        Mout_np = Mout
    else:
        assert False, "Input type invalid"
    
    # find the pairs
    max_ax = np.argmax(Mout.shape) 
    max_idx = np.argmax(Mout_np, axis=max_ax)
    max_val = np.max(Mout_np, axis=max_ax)
    if max_ax == 0:    
        max_pairs = list(zip(max_idx, np.arange(len(max_idx)), max_val))
    else:
        max_pairs = list(zip(np.arange(len(max_idx)), max_idx, max_val))
    max_pairs = [pair for pair in max_pairs if pair[2] >= thres]
    
    return Mout, max_pairs

def compute_best_matches(X, Y):
    # compute best matches between features in X and Y
    
    similarity_matrix = X @ Y.transpose(1,0)
    similarity_matrix, _ = suppress_non_max_in_matrix(similarity_matrix)
    max_sim_val, max_sim_arg = torch.max(similarity_matrix, dim=1) # find max
    max_sim_list = list(zip(np.arange(len(max_sim_arg)), max_sim_arg.cpu().numpy(), max_sim_val.cpu().numpy()))
    similarity_matrix = similarity_matrix.cpu().numpy()
            
    return max_sim_list, similarity_matrix

def remove_small_bboxes_xyxy(bboxes, thres, axis='y'):
    # bboxes : n x (4+) : bboxes, each row [x_tl, y_tl, x_br, y_br]
    # thres : threshold for removing
    # axis : remove either 'x' or 'y' axis
    
    if len(bboxes)==0:
        return bboxes
    
    if axis == 'x':
        small_detection = np.where(bboxes[:,2]-bboxes[:,0] < thres)[0]
    elif axis == 'y':
        small_detection = np.where(bboxes[:,3]-bboxes[:,1] < thres)[0]
    else:
        assert False, "Unknown axis"
    bboxes = np.delete(bboxes, small_detection, axis=0)
    return bboxes
                        
    
def list_of_member_to_group_assignment(group):
    # Convert membership list to group assignment
    # e.g., [[0,1], [1,2], [3]] -> [0, 0, 1, 1, 2]
    
    n = len(set([i for j in group for i in j]))
    out = np.zeros(n, dtype=np.long)
    for i, g in enumerate(group):
        out[g] = i
    return out
        

def list_to_affinity_matrix(group, list_member=False):
# Convert group list (e.g., [0, 0, 1, 1, 2]) to affinity matrix 
# Can also input membership list (e.g., [[0,1], [1,2], [3]])

    if list_member:
        group = list_of_member_to_group_assignment(group)
    
    unique_idx = set(group)
    aff = np.zeros((len(group), len(group)))
    group = np.array(group)
    
    for i in unique_idx:
        idx = np.where(group==i)[0]
        aff[np.ix_(idx,idx)] = 1
    
    return aff

# compute laplacian matrix
def compute_laplacian(A, type_="unnormalized"):
    A = A - np.diag(np.diag(A))
    d = np.sum(A, axis=0)
    
    if type_ == "unnormalized":
        L = np.diag(d) - A
    
    if type_ == "normalized":   
        d_ = 1/np.sqrt(d)
        L = np.eye(len(d)) - (np.diag(d_) @ A @ np.diag(d_))
        
    return L

# compute pairwise difference
def pw_diff_torch(X):
    Y = X.view(X.shape[0], 1, -1)
    Y = Y-Y.transpose(0,1)
    return Y

# reshape pairwise tensor (n x n x f) to 
# feature matrix (n^2 x f) and back
def reshape_pw_tensor(X, forward):
    if forward==1:
        Y = X.view(X.shape[0] ** 2, -1)
    elif forward==-1:
        s = np.sqrt(X.shape[0]).astype(np.int)
        Y = X.view(s, s, -1)
    else:
        assert False, "Unknown reshape direction"
        
    return Y



def vectorize_off_diag(A, k=1):
    """ Given a square matrix A, returns vectorized off-diagonal entries."""
    
    assert A.shape[0] == A.shape[1]
    
    n            = A.shape[0]
    idx_y, idx_x = np.triu_indices(n, k=k)
    out          = torch.cat((A[idx_y, idx_x], A[idx_x, idx_y]))
    return out

