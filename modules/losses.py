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

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn

from modules import utils




class LossClustering(ABC):
    """
    Abstract class for the losses used with GraphClusterer
    """
    
    @abstractmethod
    def __init__(self, loss_input_keys):
        """
        Initializer of this loss class. Requires to set keywords as input.
        
        INPUT
        -----------------------------
        loss_input_keys : list of strings   - list of keys (i.e., names) of the argument of this loss.
                                              See how this is used in __call__() of LossCollection.
        """
        self.loss_input_keys = loss_input_keys
        pass
    
    @abstractmethod
    def __call__(self):
        """
        Abstract method for the computing loss.
        
        INPUT
        -----------------------------
        """
        pass
    
    
class LossCollection:
    """
    Class for collecting and computing multiple losses of GraphClusterer
    """
    
    def __init__(self):
        self.dict_losses = {}
        self.data_keys_set = set() # set of data keys to read for evaluating, 
                                   # e.g., keywords for ground truths, mask indices. 
                                   # Will be obtained from loss.loss_input_keys
        
    def add_loss(self, name, loss, weight):
        """
        Add new loss to this object.
        
        INPUT
        -------------------
        name            : string            - name of the loss
        loss            : LossClustering    - a loss object
        weight          : float             - weight for this loss (i.e., hyperparameter lambda's)
        """
        
        assert np.isscalar(weight)
        assert isinstance(loss, LossClustering)
        
        if name in self.dict_losses:
            print("[{}] is already in dict_losses and will be overwritten.".format(name))
        self.dict_losses[name] = {"loss": loss,
                                 "weight": weight,
                                 "loss_input_keys": loss.loss_input_keys}
        self.data_keys_set |= set(loss.loss_input_keys)
        
    def get_data_dict(self, output_node, output_edge, sample):
        """
        Prepare data dictionary for loss evaluation
        
        INPUT
        -----------------------------
        output_node   : n x c matrix        - multiclass prediction (logit)
        output_edge   : n x n matrix        - prediction matrix (logit)
        sample        : Sample object       - contains (key, value) which will be used to evaluate the losses
        
        OUTPUT
        ----------------------------
        dict_output   : dictionary          - dictionary essential data for evaluating losses 
        
        """
        
        dict_output = {}
        for it_key in self.data_keys_set:
            if it_key == "output_node":
                dict_output[ "output_node"] = output_node
            elif it_key == "output_edge":
                dict_output[ "output_edge"] = output_edge
            else:
                dict_output[it_key] = getattr(sample, it_key).to(output_node.device) 
        
        return dict_output
        
        
    def __call__(self, output_node, output_edge, sample):
        """
        Compute the combination of the loss in this object.
        
        INPUT
        -----------------------------
        output_node   : n x c matrix        - multiclass prediction (logit)
        output_edge   : n x n matrix        - prediction matrix (logit)
        sample        : Sample object       - contains (key, value) which will be used to evaluate the losses
        
        OUTPUT
        ----------------------------
        total         : torch scalar        - total loss
        loss_out      : dictionary          - contains the value of each loss term weighted by lambda
        """
        
        loss_out = {}
        total = torch.zeros(1, device=output_node.device)
        
        # prepare dictionary of essential data for computing loss
        dict_est = self.get_data_dict(output_node, output_edge, sample)
        
        for key_loss in self.dict_losses:
            
            # if there is a loss with weight = 0, continue
            if self.dict_losses[key_loss]['weight'] == 0.:
                loss_out[key_loss] = torch.zeros(1, device=output_node.device)
                continue
            
            # get the parameters for this loss
            loss_input = [dict_est[i] for i in self.dict_losses[key_loss]['loss_input_keys']]
            
            # evaluate the loss
            loss_out[key_loss] = self.dict_losses[key_loss]['loss'](*loss_input)
            
            # multiply lambda
            loss_out[key_loss] = self.dict_losses[key_loss]['weight'] * loss_out[key_loss] 
            
            # sum total loss
            total += loss_out[key_loss]
            
        return total, loss_out
        
    def get_list_losses(self):
        return list(self.dict_losses.keys())
        
        
class SemisupervisedGroupClusteringLoss(LossClustering):
    """
    Computes the grouping loss by comparing affinity matrix.
    The diagonal values are not included in the loss.
    """
    
    def __init__(self, loss_input_keys, balance=False, fl_gamma=0., max_clamp=1.):
        """
        INPUT
        ---------------------------------------------------------
        loss_input_keys : list of strings   - list of keys (i.e., names) of the argument of this loss.
                                              See how this is used in __call__() of LossCollection.
        balance         : boolean           - Balance the weight of 0's and 1's in target. 
                                              If there is only one label, then this is ignored.
                                              [balance] can be used in conjunction with [group_idx], where
                                              the value in [group_idx] will be used for the balance.
        fl_gamma        : float             - gamma value if using focal loss
        max_clamp       : float             - value to clamp input to BCEloss
        """
        
        # check for 3 keys
        assert len(loss_input_keys) == 3, "Need 3 keys for [group_value, group_target, group_idx]."
        assert all([isinstance(i, str) for i in loss_input_keys]), "Need 3 keys for [group_value, group_target, group_idx]."
        
        self.loss_input_keys = loss_input_keys
        self.balance = balance
        self.bce_loss = nn.BCELoss(reduce=False)
        self.fl_gamma = fl_gamma
        self.max_clamp = max_clamp
        
        print("Initialized {} with {} as {}. Note that the order is important, so check before proceed!".format("SemisupervisedGroupClusteringLoss", loss_input_keys, "[group_value, group_target, group_idx]"))
        
        
    def __call__(self, group_value, group_target, group_idx):
        """
        INPUT
        ---------------------------------------------------------
        group_value    : n x n matrix - prediction matrix (logit)
        group_target   : n x n matrix - target matrix (binary)
        group_idx      : n vector     - an np.array of index of elements to compute the loss from
        """
        
        value = group_value
        target = group_target
        subset = group_idx
        
        
        assert value.shape == target.shape, "[value] and [target] must be square matrices of the same size."
        assert value.shape[0] == value.shape[1], "[value] and [target] must be square matrices of the same size."
        assert value.dim() == 2 and target.dim() == 2, "[value] and [target] must be 2D."
#         assert ((target == 0) | (target == 1)).all(), "[target] needs to be binary."
        assert ((target == -1) | ((target <= 1) & (target >= 0))).all(), "[target] needs to be either in [0,1] or ==-1."

        device = value.device
        n = value.shape[0]


        # consider subset
        if subset is not None and len(subset) > 1:
            
            if isinstance(subset, torch.Tensor):
                subset = subset.detach().cpu().numpy()
            
            idx = np.ix_(subset, subset)
            value = value[idx]
            target = target[idx]
        elif subset is not None and len(subset) <= 1: # if fewer than 2 persons, return 0 as no grouping needed
            return torch.zeros(1, device=device)


        value_sigmoid = torch.sigmoid(value)
        value_sigmoid_symm = (value_sigmoid+value_sigmoid.T)/2
        
        # get off diagonal values
        value_  = utils.vectorize_off_diag(value_sigmoid_symm)
        target_ = utils.vectorize_off_diag(target)


        # compute loss
        value_ = torch.clamp(value_, max=self.max_clamp)
        target_nonneg = torch.clamp(target_, min=0)# deal with target_==-1, which means do not compute loss of these entries

        if self.fl_gamma > 0:
            loss_element = (target_nonneg*((1-value_)**self.fl_gamma) + (1-target_nonneg)*(value_**self.fl_gamma))*self.bce_loss(value_, target_nonneg)
        else:
            loss_element = self.bce_loss(value_, target_nonneg)
            
        target_valid = target_>=0 # deal with target_==-1, which means do not compute loss of these entries
        loss_element = loss_element*target_valid # deal with target_==-1, which means do not compute loss of these entries
        
        # compute balance weight
        if self.balance:

            class_sum   = len(target_)
            class0_ = target_valid*(target_<0.5).float()
            class1_ = target_valid*(target_>=0.5).float()
            class0_sum  = class0_.sum()
            class1_sum  = class1_.sum()

            # only weight if there is both 0 and 1
            if class0_sum > 0 and class1_sum > 0:

                # compute weight
                weight = class_sum/2*(class0_/class0_sum + class1_/class1_sum)

                # compute loss
                loss_element = weight*loss_element

                # mean
        loss = loss_element.sum()/target_valid.sum()

        return loss




class SemisupervisedGroupTypeLoss(LossClustering):
    """
    Computes the group activity loss by comparing node vectors.
    """
    
    def __init__(self, loss_input_keys, balance=False):
        """
        INPUT
        ---------------------------------------------------------
        loss_input_keys : list of strings   - list of keys (i.e., names) of the argument of this loss.
                                              See how this is used in __call__() of LossCollection.
        balance         : boolean           - Balance the weight of 0's and 1's in target. 
                                              If there is only one label, then this is ignored.
                                              [balance] can be used in conjunction with [node_idx], where
                                              the value in [node_idx] will be used for the balance.
        """
        
        # check for 3 keys
        assert len(loss_input_keys) == 3, "Need 3 keys for [node_value, node_target, node_idx]."
        assert all([isinstance(i, str) for i in loss_input_keys]), "Need 3 keys for [node_value, node_target, node_idx]."
        
        self.loss_input_keys = loss_input_keys
        self.balance = balance
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        print("Initialized {} with {} as {}. Note that the order is important, so check before proceed!".format("SemisupervisedGroupTypeLoss", loss_input_keys, "[node_value, node_target, node_idx]"))
        
        
        
    def __call__(self, node_value, node_target, node_idx):
        """
        Computes the group type loss.

        INPUT
        ---------------------------------------------------------
        node_value    : n x c matrix - multiclass prediction (logit)
        node_target   : n vector     - target class vector
        node_idx      : n vector     - an np.array of index of elements to compute the loss from
        """
        
        value = node_value
        target = node_target
        subset = node_idx
        
        assert value.shape[0] == target.shape[0], "Number of data in [value] and [target] must be the same."
        assert value.dim() == 2, "[value] must be 2D."
        assert target.dim() == 1 and len(subset.shape) == 1, "[target] and [subset] must be 1D."

        device = value.device
        n_classes = value.shape[1]

        # select subset
        if subset is not None and len(subset) > 0:
            
            if isinstance(subset, torch.Tensor):
                subset = subset.detach().cpu().numpy()
            
            target = target[subset]
            value = value[subset]
        elif subset is not None and len(subset) == 0: # if no subset, return 0
            return torch.zeros(1, device=device)

        # cross entropy
        loss_element = self.ce_loss(value, target)

        if self.balance:        
            # compute weight
            class_sum = torch.zeros(n_classes, device=device)
            weight = torch.zeros_like(target, dtype=torch.float, device=device)
            valid_classes = 0
            for i in range(n_classes):
                class_sum[i] = (target==i).sum()

                if class_sum[i] > 0: 
                    weight[target==i] = 1/class_sum[i]
                    valid_classes += 1


            # compute weight vector
            weight = weight*class_sum.sum()/valid_classes

            # weight the loss
            loss_element = weight*loss_element

        loss = loss_element.mean()
        return loss
    
    

class QueueEndLoss(LossClustering):
    """
    Computes the queue end loss.
    """
    
    def __init__(self, loss_input_keys):
        """
        INPUT
        ---------------------------------------------------------
        loss_input_keys : list of strings   - list of keys (i.e., names) of the argument of this loss.
                                              See how this is used in __call__() of LossCollection.
        """
        # check for 2 keys
        assert len(loss_input_keys) == 2, "Need 2 keys for [node_value, node_target, node_idx]."
        assert all([isinstance(i, str) for i in loss_input_keys]), "Need 2 keys for [node_value, node_target]."
        
        print("Initialized {} with {} as {}. Note that the order is important, so check before proceed!".format("QueueEndLoss", loss_input_keys, "[node_value, node_target]"))
        self.loss_input_keys = loss_input_keys
        
        
    def __call__(self, node_value, node_target):
        """
        Computes the queuing loss.

        INPUT
        ---------------------------------------------------------
        node_value    : n vector     - prediction (logit)
        node_target   : n vector     - binary vector indicating queue end (with only one 1 and all other 0's)
        """
        
        node_value = node_value.squeeze()
        node_target = node_target.squeeze()
        
        # check dimension and that there is only one queue end
        assert node_value.dim() == 1
        assert all(node_target*(1-node_target) == 0) and node_target.sum() == 1 and node_target.dim() == 1
        
        # get index of queue end
        idx_queue_end = node_target.nonzero().squeeze()
        
        # cross entropy loss
        loss = -node_value[idx_queue_end]+torch.logsumexp(node_value, dim=0)
        
        return loss
    
    

class LossL1(LossClustering):
    """
    L1 loss over affinity matrix
    """
    
    def __init__(self, loss_input_keys):
        """
        Initializer of this loss class. Requires to set keywords as input.
        
        INPUT
        -----------------------------
        loss_input_keys : list of strings   - list of keys (i.e., names) of the argument of this loss.
                                              See how this is used in __call__() of LossCollection.
        """
        
        # check for 3 keys
        assert len(loss_input_keys) == 2, "Need 3 keys for [group_value, group_target]."
        assert all([isinstance(i, str) for i in loss_input_keys]), "Need 3 keys for [group_value, group_target]."
        
        self.loss_input_keys = loss_input_keys
        pass
    
    def __call__(self, group_value, group_target):
        """
        Abstract method for the computing loss.
        
        INPUT
        -----------------------------
        group_value    : n x n matrix - prediction matrix (logit)
        group_target   : n x n matrix - target matrix (binary)
        """
        triu = torch.triu(group_target, 1)
        triu_1_idx = torch.where(triu)
        if len(triu_1_idx[0]) == 0: # if no entries with 1 in triu, return loss=0.
            return torch.zeros(1, device=group_target.device)
        
        group_val_sel_sigmoid = group_value[triu_1_idx].sigmoid()
        loss = group_val_sel_sigmoid.mean()
        
        return loss
