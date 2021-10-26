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
from shapely.geometry import Polygon
import torch

from modules.utils import list_of_member_to_group_assignment, list_to_affinity_matrix
from modules.datautils import Sample

def extract_block_info(img_anno_dict, id_cat_block, cat_relabel=None, thres_iou_inside=0.9):
    """
    Extract the ground truth block information from the 
    preprocessed json file.
    
    Here, we consider a segment (4-pt bbox annotation) to belong to
    a block if their 90% of the segment is inside the block.
    The parameter 90% is adjustable with 'thres_iou_inside'.

    We also do additional processing:
    1. We remove nested blocks and use only the top-level block
    2. We put any segment/bbox which are not inside any block to its own block

    INPUT
    ------
    img_anno_dict    : list of annotations   - list of annotations from the COCO json file
    id_cat_block     : int                   - the category id of 'block' object
    cat_relabel      : dict                  - dictioanry to relabel cat_id of the bboxes  
    thres_iou_inside : float                 - threshold of the ratio of iou/area to consider a box is inside another box
    
    OUTPUT
    ------
    dict_output      : dict                  - dictionary of output (see end of the code)
    """
    
    # list all annos of that image
    list_anno_img_block = []
    list_anno_img_nonblock = []
    for anno_tmp in img_anno_dict:

        bbox = np.array(anno_tmp['bbox'])
        segm = np.array(anno_tmp['segmentation']).reshape(4,2)
        cat_id = anno_tmp['category_id']
        dict_tmp = {'bbox': bbox, 'segm': segm, 'cat_id': cat_id}

        if anno_tmp['category_id'] == id_cat_block:
            list_anno_img_block.append(dict_tmp)
        else:
            list_anno_img_nonblock.append(dict_tmp)

    # convert segment to polygon
    pg_block = [Polygon(i['segm']) for i in list_anno_img_block]
    pg_nonblock = [Polygon(i['segm']) for i in list_anno_img_nonblock]

    # remove blocks which are inside other box
    block_to_remove = set()
    for it_block1 in range(len(pg_block)):
        for it_block2 in range(it_block1+1, len(pg_block)):
            pg1 = pg_block[it_block1]
            pg2 = pg_block[it_block2]

            int_area = pg1.intersection(pg2).area
            if int_area / max(1, pg1.area) > thres_iou_inside:
                block_to_remove.add(it_block1)
            elif int_area / max(1,pg2.area) > thres_iou_inside:
                block_to_remove.add(it_block2)
    pg_block = [blk for it, blk in enumerate(pg_block) if it not in block_to_remove]

    # assign data to block
    list_group_members = [[] for i in range(len(pg_block))]
    for it_data in range(len(pg_nonblock)):
        assign = False

        for it_block in range(len(pg_block)):

            pgd = pg_nonblock[it_data]
            pgb = pg_block[it_block]

            int_area = pgd.intersection(pgb).area

            if int_area / max(1, pgd.area) > thres_iou_inside:
                assign = True
                list_group_members[it_block].append(it_data)
                break

        if not assign:
            list_group_members.append([it_data])

    node_group_id = list_of_member_to_group_assignment(list_group_members)
    bbox = np.array([i['bbox'] for i in list_anno_img_nonblock])
    segm = np.array([i['segm'] for i in list_anno_img_nonblock])
    cat_id = np.array([i['cat_id'] for i in list_anno_img_nonblock])
    
    if cat_relabel is not None:
        cat_id = [cat_relabel[i] for i in cat_id]
    
    dict_output = {}
    dict_output['bbox']               = bbox   # list of bboxes
    dict_output['segm']               = segm   # list of segments (4-point bbox)
    dict_output['cat_id']             = cat_id # category of each box
    dict_output['node_group_id']      = node_group_id # node group id, e.g., [0,0,1,1,2]
    dict_output['list_group_members'] = list_group_members # member of each group, e.g., [[0,1],[2,3],[4]]
    
    return dict_output



def datadict2sample(datadict, n_categories, dim_edge_feature=64):
    """
    Convert datadict to Sample object for training model.
    
    INPUT
    ------
    datadict   : dictionary      - processed data in dictionary form.
                                   Contains following keys dict_keys(['bbox', 'segm', 'cat_id', 'node_group_id', 'list_group_members', 'mask', 'img_rgb_scaled'])
    n_categories : int           - number of categories
    dim_edge_feature : int       - dimension of edge feature (all zero tensor)
    
    OUTPUT
    ------
    sampke     : Sample          - a Sample object
    """
    img_rgb_scaled = datadict['img_rgb_scaled']
    img_rgb_scaled = img_rgb_scaled/128-1 # normalize to -1, 1

    masks              = datadict['mask']
    cat_id             = datadict['cat_id']
    list_group_members = datadict['list_group_members']
    node_group_id      = datadict['node_group_id']

    # compute the shape size
    mask_shape     = masks[0].shape
    node_feat_shape = list(mask_shape)
    node_feat_shape = [n_categories] + node_feat_shape # set the shape to the number of categories

    # generate node feature
    node_features = []
    for it_mask in range(len(masks)):

        mask = masks[it_mask]

        # node feature for categories and mask
        node_feature_cat = torch.zeros(node_feat_shape)

        # set the mask in the correct category
        node_feature_cat[[cat_id[it_mask]], :, :] = mask

        # put together with image
        node_feature = torch.cat([img_rgb_scaled, node_feature_cat], dim=0)
        node_features.append(node_feature[None,])

    # convert node_features to torch 
    node_feature_torch = torch.cat(node_features, dim=0)
    
    # get gt_aff_mat
    gt_aff_mat = torch.from_numpy(list_to_affinity_matrix(node_group_id)).to(torch.float)

    # edge feature
    n_data_tmp = node_feature_torch.shape[0]
    edge_feature_torch = torch.zeros(n_data_tmp, n_data_tmp, dim_edge_feature)
    
    # details
    details_dict = {}
    details_dict['filename'] = datadict['filename']
    details_dict['segm'] = datadict['segm']
    details_dict['orig_segm'] = datadict['orig_segm']
    details_dict['img_id'] = datadict['img_id']
    details_dict['cat_id'] = datadict['cat_id']
    
    # add as sample
    sample = Sample(node_feature_torch, 
                    edge_feature_torch, 
                    gt_aff_mat,
                    None,
                    node_group_id,
                    list_group_members,
                    details=details_dict)
    
    return sample
