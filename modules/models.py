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
import torch
import torch.nn as nn
import torch.nn.functional as F
from .datautils import Sample
from .graph_clustering import cluster_affinity

def gen_mlp(n_in, n_hid, n_out, use_norm=False, last_linear=False):
    # generate 2-layer MLP
    
    layers = []
    
    layers.append(nn.Linear(n_in, n_hid))
    if use_norm:
        layers.append(nn.GroupNorm(4, n_hid))
    layers.append(nn.LeakyReLU(0.2))
    
    layers.append(nn.Linear(n_hid, n_out))
    if last_linear:
        return nn.Sequential(*layers)
    
    if use_norm:
        layers.append(nn.GroupNorm(4, n_out))
    layers.append(nn.LeakyReLU(0.2))
    
    return nn.Sequential(*layers)



class MultiheadedGraphAttentionLayer(nn.Module):
    """
    Graph attention layer module
    """
    
    def __init__(self, 
                 dim_node_input, 
                 dim_edge_input=0, 
                 n_heads=1, 
                 use_residual=False, 
                 use_norm=False):
        """
        dim_node_input : int         - dimension of node features
        dim_edge_input : int         - dimension of edge feature
        n_heads        : int         - number of head of GAT
        use_residual   : bool        - to use residual in GAT or not
        use_norm       : bool        - to use normalization layer or not
        """
        super(MultiheadedGraphAttentionLayer, self).__init__()
        
        if use_norm:
            assert (dim_node_input % 4 == 0) and (dim_edge_input % 4 == 0), 'If use_norm=True, need dim of features to be divisible by 4.'
        
        self.dim_node_input  = dim_node_input
        self.dim_edge_input  = dim_edge_input
        self.dim             = dim_node_input + dim_edge_input
        self.use_residual    = use_residual
        self.n_heads         = n_heads
        
        # attention mlp
        self.mlp_att = gen_mlp(2*self.dim, self.dim, self.n_heads, use_norm=False) # for attention, don't use batch norm
        
        # node feature transform mlp
        self.mlp_node_feat = gen_mlp(2*self.n_heads*self.dim, self.dim_node_input, self.dim_node_input, use_norm=use_norm)
    
        # edge feature transform mlp
        if self.dim_edge_input > 0:
            self.mlp_edge_feat = gen_mlp(2*self.dim, self.dim_node_input, self.dim_node_input, use_norm=use_norm)
            
            
    def forward(self, feat_node, feat_edge=None):
        """
        feat_node:    n_nodes x dim_node_input
        feat_edge:    n_nodes x n_nodes x dim_edge_input
        """
        
        assert feat_node.shape[1] == self.dim_node_input
        assert (feat_edge is None) or (feat_edge.shape[2] == self.dim_edge_input)
        
        n_nodes = feat_node.shape[0]
        
        x_pw0 = feat_node.unsqueeze(0).expand(n_nodes, -1, -1)
        x_pw1 = feat_node.unsqueeze(1).expand(-1, n_nodes, -1)
        
        # case with edge or without edge feature
        if self.dim_edge_input > 0:
            feat_edge_t = feat_edge.transpose(0, 1)
            x_pw = torch.cat([x_pw0, x_pw1, feat_edge, feat_edge_t], dim=2) # include edge
        else:
            x_pw = torch.cat([x_pw0, x_pw1], dim=2) # only node feature (no edge)
            
        # compute attention
        att = self.mlp_att(x_pw)
        att = torch.softmax(att, dim=1)
        
        # compute attended feature
        # Here, we exploit a computation trick from Eq (6) in https://arxiv.org/pdf/1710.10903.pdf
        # that we can pull W^k (i.e., mlp_node_feat in this implementation) in front of the sum_j,
        # so we can put [mlp_node_feat] after attention and concatenation
        x_att = att.unsqueeze(3) * x_pw.unsqueeze(2) # multiply attention (unsqueeze to deal with n_heads)
        x_out = torch.sum(x_att, dim=1) # attention
        x_out = x_out.view(n_nodes, 2*self.n_heads*self.dim) # concatenate (by reshape) the attended features
        x_out = self.mlp_node_feat(x_out) # transform feature
        
        if self.use_residual: # residual
            x_out = x_out + feat_node
        
        # compute edge feature
        if self.dim_edge_input > 0:
            x_pw_e_t = x_pw.view(n_nodes*n_nodes, 2*self.dim)
            x_e_out = self.mlp_edge_feat(x_pw_e_t) # transform feature
            x_e_out = x_e_out.view(n_nodes, n_nodes, self.dim_edge_input)
            
            if self.use_residual: # residual
                x_e_out = x_e_out + feat_edge
        
            return x_out, x_e_out
        
        return x_out
        
        
        
class GraphAttentionNet(nn.Module):
    """
    This model basically stack GraphAttentionLayer into a large network 
    """
    
    def __init__(self,
                 n_gat_layers, 
                 dim_node_input, 
                 dim_node_output=0,
                 dim_edge_output=0,
                 dim_edge_input=0, 
                 n_heads=1, 
                 use_residual=False, 
                 use_norm=False,
                 node_feature_extractor=None,
                 edge_feature_extractor=None,
                 no_node_input=False):
        """
        dim_node_input         : int         - dimension of node features for internal processing.
                                               The input feature must have this dimension, unless
                                               node_feature_extractor is specified (see
                                               node_feature_extractor).
        dim_node_output        : int         - dimension of node output
        n_gat_layers           : int         - number of GAT layers
        dim_edge_output        : int         - dimension of edge output
        dim_edge_input         : int         - dimension of edge feature
                                               Details similar to 'dim_node_input'.
        n_heads                : int         - number of head of GAT
        use_residual           : bool        - to use residual in GAT or not
        use_norm               : bool        - to use normalization layer or not
        node_feature_extractor : nn.Module   - transform node feature to size [n_nodes x dim_node_input]
        edge_feature_extractor : nn.Module   - transform edge feature to size [n_nodes x n_nodes x dim_edge_input]
        no_node_input          : bool        - indicator to tell that the network will not
                                               receive node input, and thus will use zero
                                               vectors as the first layer's node features.
                                               Its size will be [n_nodes x dim_node_input].
                                               Note: the input to feat_node of forward() 
                                               function must be None, otherwise error will occur.
                                               Also, if set to True, then the model will ignore
                                               node_feature_extractor.
        """ 
        super(GraphAttentionNet, self).__init__()
        
        self.dim_node_input  = dim_node_input    
        self.n_gat_layers    = n_gat_layers
        self.dim_edge_input  = dim_edge_input  
        self.n_heads         = n_heads     
        self.use_residual    = use_residual
        self.use_norm        = use_norm    
        self.dim_node_output = dim_node_output
        self.dim_edge_output = dim_edge_output
        self.node_feature_extractor = node_feature_extractor
        self.edge_feature_extractor = edge_feature_extractor
        self.no_node_input   = no_node_input
        
        gat_layers = []
        for i in range(self.n_gat_layers):
            gat = MultiheadedGraphAttentionLayer(self.dim_node_input, 
                                                 self.dim_edge_input, 
                                                 use_norm=self.use_norm, 
                                                 use_residual=self.use_residual,
                                                 n_heads=self.n_heads)
            gat_layers.append(gat)
        self.gat_layers = nn.Sequential(*gat_layers) # put in sequential (since pytorch does not compute gradient for parameters in list)
        
        # layer for the node output
        self.node_tf = gen_mlp(self.dim_node_input, self.dim_node_input, self.dim_node_output, use_norm=self.use_norm, last_linear=True) # last layer is linear
        
        # layer for the edge output
        if self.dim_edge_input > 0 and self.dim_edge_output > 0:
            self.edge_tf = gen_mlp(self.dim_edge_input, self.dim_edge_input, self.dim_edge_output, use_norm=self.use_norm,last_linear=True) # last layer is linear
       
    
    def parse_sample(self, sample):
        """
        Parse Sample object into input for the network.
        
        INPUT
        ---------
        sample    : Sample object  - input Sample object
        
        OUTPUT
        ---------
        node_feature
        edge_feature
        """
        
        # get device
        device = self.node_tf[0].weight.device
        
        # parse the Sample object
        if sample.node_feature is not None:
            node_feature = sample.node_feature.squeeze(0).to(device)
        else:
            node_feature = None

        if sample.edge_feature is not None:
            edge_feature = sample.edge_feature.squeeze(0).to(device)
        else:
            edge_feature = None
            
        return node_feature, edge_feature
    
    
    def parse_forward_input(self, feat_node, feat_edge=None, sample=None):
        """
        Parse different cases of input parameters into
        those processable by forward().
        """
        
        # first condition : one of the input is not None
        # second condition : either sample or one of feat_node, feat_edge must be None
        assert (feat_node is not None or feat_edge is not None or sample is not None) and (sample is None or (feat_node is None and feat_edge is None)), "Invalid input type: feat_node ({}), feat_edge({}), sample ({})".format(type(feat_node), type(feat_edge), type(sample))
        
        # if feat_node is a Sample object, parse it
        if feat_node is None and feat_edge is None and isinstance(sample, Sample):
            feat_node, feat_edge = self.parse_sample(sample)
            
        if self.no_node_input:
            assert feat_node is None, "feat_node must be None to use with self.no_node_input=True."
            feat_node = torch.zeros(feat_edge.shape[0], self.dim_node_input, device=feat_edge.device)
        
        # if there are additional feature extractors, run them
        if self.node_feature_extractor is not None and not self.no_node_input:
            feat_node = self.node_feature_extractor(feat_node)
            
        if self.edge_feature_extractor is not None and feat_edge is not None:
            feat_edge = self.edge_feature_extractor(feat_edge)
        
        return feat_node, feat_edge
    
        
    def forward(self, feat_node, feat_edge=None, sample=None):
        """
        feat_node:    n_nodes x dim_node_input (unless node_feature_extractor is given)
        feat_edge:    n_nodes x n_nodes x dim_edge_input (unless edge_feature_extractor is given)
        sample: use this when want to directly input Sample object
        Note: feat_node can also be a Sample object.
        """
        
        feat_node, feat_edge = self.parse_forward_input(feat_node, feat_edge, sample)
            
        x_node = feat_node
        x_edge = feat_edge
        
        if x_node is not None:
            n_nodes = x_node.shape[0]
        else:
            n_nodes = x_edge.shape[0]
            
        # pass data through GATs
        for i in range(self.n_gat_layers):
            if feat_edge is not None:
                x_node, x_edge = self.gat_layers[i](x_node, x_edge)
            else:
                x_node = self.gat_layers[i](x_node)
                
        # final layer for node
        node_output = self.node_tf(x_node)
                
        # return
        if self.dim_edge_input > 0 and self.dim_edge_output > 0:
            edge_output = self.edge_tf(x_edge.view(n_nodes*n_nodes, self.dim_edge_input))
            edge_output = edge_output.view(n_nodes, n_nodes, self.dim_edge_output)
            return node_output, edge_output.squeeze()
        
        return node_output
    
    
    def infer_clusters(self, 
                       feat_node, 
                       feat_edge=None, 
                       sample=None,
                       thres_aff=0.5,
                       thres_cluster=0.1,
                      ):
        """
        Run the model but unlike forward(), this function
        also performs clustering and assigning labels to groups.
        
        INPUT
        -------
        feat_node :    n_nodes x dim_node_input (unless node_feature_extractor is given)
        feat_edge :    n_nodes x n_nodes x dim_edge_input (unless edge_feature_extractor is given)
        sample    : use this when want to directly input Sample object
        Note: feat_node can also be a Sample object.
        thres_aff : float       - threshold of affinity matrix
        thres_cluster : float   - threshold for cuting the eigenvalue of the normalized laplacian matrix (spectral clustering)
        
        OUTPUT
        -------
        output             : dictionary     - dictionary of outputs
         - n_groups        : int            - number of groups
         - group_idx       : np.array       - group index of each node
         - pred_class      : np.array       - class label of each node
        """
        
        # pass to forward
        with torch.no_grad():
            node_output, edge_output = self.forward(feat_node, feat_edge, sample)
            edge_output_sigmoid = edge_output.sigmoid().detach().cpu().numpy()
            edge_output = edge_output.detach().cpu().numpy()
            node_output_prob = torch.softmax(node_output, dim=1).detach().cpu().numpy()

        if node_output_prob.shape[1] > 0:
            node_output_class = np.argmax(node_output_prob, axis=1)
        n_nodes = node_output_prob.shape[0]
           
        # compute affinity matrix by thresholding at 0.5
        aff_mat       = (edge_output_sigmoid + edge_output_sigmoid.T)/2
        aff_mat_thres = np.maximum(0,aff_mat-thres_aff)/(1-thres_aff)
        
        # clustering
        group_id = cluster_affinity(aff_mat_thres, thres_cluster=thres_cluster)
        n_groups  = np.max(group_id)+1 
        
        # assign class label by majority vote of each group
        if node_output_prob.shape[1] > 0:
            pred_class  = -np.ones(n_nodes, dtype=np.long)
            for it_group in range(n_groups):
                idx_tmp = np.where(group_id==it_group)[0]
                cls_tmp, count_tmp = np.unique(node_output_class[idx_tmp], return_counts=True)
                pred_class[idx_tmp] = cls_tmp[np.argmax(count_tmp)]
        else:
            pred_class = None
            
        output = {}
        output['n_groups']    = n_groups
        output['group_id']    = group_id
        output['pred_class']  = pred_class  
        output['aff_mat']     = aff_mat
        
        
        return output
        
