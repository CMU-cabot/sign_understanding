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

import torch
from torch.utils.data import DataLoader
import numpy as np 

class Sample:
    """
    Class for each sample of training data
    """
    def __init__(self,
                 node_feature,
                 edge_feature,
                 gt_aff_mat,
                 gt_node_class,
                 node_group_id,
                 list_group_members,
                 list_idx_label_group=None,
                 list_idx_label_class=None,
                 details=None):
        """
        See example below
        
        NOTE: the order of members in list_group_members is important (e.g., for
        queues, the order might be used to evaluate the order/ranking of members). 
        """
        
        # say we have n_nodes=5
        if node_feature is not None:
            n_nodes = node_feature.shape[0]
        else:
            n_nodes = edge_feature.shape[0]
        
        # torch.tensor
        self.node_feature       = node_feature       # n_nodes x ...
        self.edge_feature       = edge_feature       # n_nodes x n_nodes x ...
        self.gt_aff_mat         = gt_aff_mat          # n_nodes x n_nodes 
        self.gt_node_class      = gt_node_class      # n_node np.array (int)
        
        # list
        self.node_group_id      = node_group_id      # e.g., [0,0,1,1,2]
        self.list_group_members = list_group_members # e.g., [[0,1],[2,3],[4]]
        self.details            = details            # dictionary of other details (not used by training/testing, but maybe visualization)

        # torch.tensor
        # list of index (on {0,...,n_nodes-1}) that has group label
        if list_idx_label_group is not None:
            self.list_idx_label_group = list_label_group
        else:
            self.list_idx_label_group = torch.arange(n_nodes)
        
        # torch.tensor
        # list of index (on {0,...,n_nodes-1}) that has class label
        if list_idx_label_class is not None:
            self.list_idx_label_class = list_label_group
        else:
            self.list_idx_label_class = torch.arange(n_nodes)

            
class GroupDataLoader(DataLoader):
    """
    A dataloader class for group data.
    Only uses collate_fn=do_nothing function to output list of samples
    instead of the default collate_fn.
    """
    
    def __init__(self, dataset, **args):
        
        if 'collate_fn' in args:
            print("GroupDataLoader: Found 'collate_fn' in args. Will be ignored.")
        do_nothing = lambda x: x
        args['collate_fn'] = do_nothing
        
        super(GroupDataLoader, self).__init__(dataset, **args)
        
        
