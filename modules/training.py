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

import os

from datetime import datetime
import tensorboardX as tb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from .losses import LossCollection
from .models import GraphAttentionNet



def train_graph_attention_net(grouping_model,
                              dataloader, 
                              loss_collection, 
                              optimizer,
                              n_epochs, 
                              path_save_training_tb, 
                              path_save_training_model, 
                              freq_save_model, 
                              device, 
                              scheduler=None,
                              freq_save_tb=20):
    """
    Train GraphClusterer model
    
    INPUT
    ---------------------------
    grouping_model             : GraphAttentionNetwork    - graph clustering model
    dataloader                 : dataloader               - dataloder (batchsize=1)
    loss_collection            : LossCollection           - collection of losses
    optimizer                  : Optimizer                - optimizer
    n_epochs                   : int                      - number of epochs to train
    path_save_training_tb      : string                   - path to save tensorboard
    path_save_training_model   : string                   - path to save training model
    freq_save_model            : string                   - frequency to save model (every n epoch)
    device                     : str/device               - device on which the model is trained
    scheduler                  : Scheduler                - step size scheduler
    freq_save_tb               : int                      - frequency to save tensorboard (every n batch)
    
    OUTPUT
    --------------------------
    grouping_model             : GraphClusterer    - trained graph clustering model
    """

#     assert isinstance(grouping_model, GraphAttentionNet)
#     assert isinstance(dataloader, DataLoader)
#     assert isinstance(loss_collection, LossCollection)
#     assert isinstance(optimizer, torch.optim.Optimizer)
#     assert isinstance(n_epochs, int)
#     assert isinstance(path_save_training_tb, str)
#     assert isinstance(path_save_training_model, str)
#     assert isinstance(freq_save_model, int)
#     assert isinstance(freq_save_tb, int)
    
    # define necessary items
    writer = tb.SummaryWriter(path_save_training_tb)
    n_data = len(dataloader)
    batch_counter = 0 # count batch for tensorboard
    balance = True #### 
    it_epoch = 0
    pbar = tqdm(total=n_epochs)

    print("Begin training")
    print("See tensorboard in [tensorboard --port xxxx --logdir {}]".format(path_save_training_tb))
    pbar.n = it_epoch
    pbar.refresh()
    while it_epoch < n_epochs:

        iter_dataloader = iter(dataloader)

        for it_batch in range(len(dataloader)): 

            # total loss
            loss = torch.zeros(1, device=device)
            
            # loss_batch
            loss_batch = {}
            for key_loss in loss_collection.get_list_losses():
                loss_batch[key_loss] = torch.zeros(1, device=device)

            samples = iter_dataloader.next() 
            
            for it_data, sample in enumerate(samples):

                # pass to model
                output_node, output_edge = grouping_model(None, sample=sample)
    
                # compute loss
                loss_tmp, dict_loss_vals = loss_collection(output_node, output_edge, sample)

                # combine loss
                for key_loss in dict_loss_vals:
                    loss_batch[key_loss] += dict_loss_vals[key_loss]
                loss += loss_tmp
                
                # check for nan
                if not (loss == loss):
                    assert False

            # divide losses by batch size
            for key_loss in loss_batch:
                loss_batch[key_loss] /= len(samples) # not divided batch_size in case last batch not complete
            loss /= len(samples)


            # optimizer
            optimizer.zero_grad()
            if loss.requires_grad: # in case the samples have no gradients, just skip
                loss.backward()
                optimizer.step()

            # record loss
            batch_counter += 1 # increment batch counter
            
            # write to tensorboard
            if batch_counter % freq_save_tb == 0:
                for key_loss in loss_batch:
                    writer.add_scalar(key_loss, loss_batch[key_loss].item(), batch_counter)
                writer.add_scalar("0_Loss", loss.item(), batch_counter)

        if scheduler is not None:
            scheduler.step()
            
        # save model
        if it_epoch % freq_save_model == 0:
            if not os.path.isdir(path_save_training_model):
                os.makedirs(path_save_training_model)
            torch.save(grouping_model, os.path.join(path_save_training_model, "model_{:07d}.pth".format(batch_counter)))

        it_epoch += 1
        pbar.n = it_epoch
        pbar.refresh()
    
    torch.save(grouping_model, os.path.join(path_save_training_model, "model_final.pth".format(batch_counter)))
    return grouping_model
