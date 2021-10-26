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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../utils')

import modules.utils as utils_fn

import cv2
import enum
import imageio
import os
from tqdm import tqdm
from types import GeneratorType




def draw_bboxes(ax, bboxes, bboxes_type="xyxy", edge_color=[[1., 0., 0., 1.]], text=None, fill_colors=None, fontsize=12, linewidth=2):
    """
    Draw bounding boxes on the axis
    Can specify fill and edge colors and text of each bbox
    Assume the format of bboxes are [x, y, width, height]
    """
    
    edge_color = np.array(edge_color)
    if edge_color.ndim == 1:
        edge_color = edge_color[None, :] # do this just for check after assert
    
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    assert len(edge_color) == 1 or len(edge_color) == len(bboxes)
    assert text is None or len(text) == len(bboxes)
    
    if len(bboxes) < 1:
        return
    
    if bboxes_type=="xyxy":
        bboxes = utils_fn.tform_bboxes_xyxy2xywh(bboxes)
    elif bboxes_type=="xywh":
        None
    else:
        assert False, "Wrong bboxes_type"
    
    edge_color = edge_color.squeeze()
    
    # draw bboxes
    for it, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        
        # select color
        if type(edge_color) is np.ndarray and (edge_color.ndim > 1) and edge_color.shape[0] >= 1:
            color_ = edge_color[it]
        else:
            color_ = edge_color
            
        # draw
        if fill_colors is None:
            ax.add_patch(plt.Rectangle([x,y], w, h, fill=False, edgecolor=color_, linewidth=linewidth))
        else:
            ax.add_patch(plt.Rectangle([x,y], w, h, facecolor=fill_colors[it], fill=True, edgecolor=color_, linewidth=linewidth))
        
        if text is not None:
            ax.text(x, y-4, "{}".format(text[it]), fontdict=dict(fontsize=fontsize, color=color_, weight='bold'))
    
    


def visualize_img_bboxes_xyxy(ax, img, bboxes, bboxes_type="xyxy", color=None, fill_colors=None, text=None, fontsize=12, linewidth=2):
    # draw image with bboxes and color-coded with group activity
    
    # draw image
    ax.imshow(img)
    
    if len(bboxes) == 0:
        return
    
    if type(color) is list:
        color = np.array(color)
        
    # draw bboxes
    for it, bbox in enumerate(bboxes):
        
        # select color
        if (type(color) is np.ndarray):
            if (len(color.shape) > 1) and color.shape[0] >= 1:
                color_ = color[it]
            else:
                color_ = color
        else:
            color_ = 'red'
            
        bbox = np.concatenate([bbox, bbox[[0]]], axis=0)
        ax.plot(bbox[:,0], bbox[:,1], color=color_, linewidth=linewidth)
        
        if text is not None:
            ax.text(bbox[0,0], bbox[0,1]+4, "{}".format(text[it]), fontdict=dict(fontsize=fontsize, color=color_, weight='bold'))



def visualize_groups(ax, img, bboxes, group_labels, group_types, idx_qe=[], bboxes_type="xyxy", color=None, fill_colors=None, text=None, fontsize=12, linewidth=2):



    # visualization
    ax.cla()
    ax.axis('off')

    n_detection = len(bboxes)

    if n_detection > 1:

        # relabel group labels so that they start from 0 and 
        # let label 0 be group of individuals, so they can be
        # colored using same color
        lab, cou = np.unique(group_labels, return_counts=True)
        single_idxs = [lab[j] for j, it in enumerate(cou) if it==1]
        group_idxs = [lab[j] for j, it in enumerate(cou) if it!=1]
        dict_relabel = {i:0 for i in single_idxs}
        dict_relabel.update({i:j+1 for j,i in enumerate(group_idxs)})
        labels_ = [dict_relabel[i] for i in group_labels]

        # select colors
        colors = plt.cm.hsv(np.linspace(0, 1, np.max(labels_)+1))
        colors = np.vstack([[[0.75, 0.75, 0.75, 1]], colors]) # color of individuals
        colors = colors[labels_]

        # visualize
        if fill_colors is not None:
            bboxes_fill_colors = [fill_colors[j] for j in group_types]
        else:
            bboxes_fill_colors = None
            
        visualize_img_bboxes_xyxy(ax, 
                                  img, 
                                  bboxes, 
                                  text=[i for i in range(len(bboxes))], color=colors, 
                                  fill_colors=bboxes_fill_colors)

        # visualizing queue end
        if len(idx_qe)>0:
            visualize_img_bboxes_xyxy(ax, img, utils_fn.scale_bbox_xyxy(bboxes[idx_qe], 1.1), color=[1,1,0,1], linewidth=linewidth)


    else:
        visualize_img_bboxes_xyxy(ax, img, bboxes, text=[i for i in range(len(bboxes))], color=[1.,1.,1., 1.], linewidth=linewidth)

        
        


def write_video(generator_bgr_bbox: GeneratorType,
                path_vid_out: str):
    """
    Write video for visualizing result of grouping and queue end model
    
    INPUT
    -------------------
    generator_bgr_bbox   : GeneratorType        - generator that returns dictionary with following key
        - img                : np.array             - RGB uint8 image
        - bboxes             : np.array             - 4 x n array indicating bboxes
        - frame_id           : int                  - frame id
        - total_frames       : int                  - total number of frames
        - node_clusters      : np.array             - array of cluster index of each bbox
        - node_group_type    : np.array             - array of person-wised group activity
        - queue_end_idx      : np.array             - array of person indicated as queue end
        - group_color_dict   : dict                 - dictionary assigning group activity label to color
    path_vid_out         : str                  - output file name
    """
    
    
    # set vid writer
    vid_writer = imageio.get_writer(path_vid_out, fps=10)
    
    # figure
    fig, axs = plt.subplots(2, 2, figsize=(25, 18))
    
    pbar = None
    
    # loop
    for output_dict in generator_bgr_bbox:
        
        # read data
        img_rgb          = output_dict["img"]
        bboxes           = output_dict["bboxes"]
        frame_id         = output_dict["frame_id"]
        total_frames     = output_dict["total_frames"]
        node_clusters    = output_dict["node_clusters"]
        node_group_type  = output_dict["node_group_type"]
        queue_end_idx    = output_dict["queue_end_idx"]
        group_color_dict = output_dict["group_color_dict"]
        aff_mat          = output_dict["aff_mat"]
        
        if "plot_gt" in output_dict:
            plot_gt          = output_dict["plot_gt"]
            gt_node_clusters = output_dict["gt_node_clusters"]
            gt_node_group_type = output_dict["gt_node_group_type"]
            gt_queue_end_idx = output_dict["gt_queue_end_idx"]
        else:
            plot_gt = False
            
        # tqdm
        if pbar is None:
            pbar = tqdm(range(total_frames), desc='This progress bar lowerbounds the progress.')
        else:
            pbar.n = frame_id
            pbar.refresh()
            
        #################
        # VISUALIZATION #
        #################
        
        # clear output
        axs[0,0].cla()
        axs[0,1].cla()
        axs[1,0].cla()
        axs[1,1].cla()
        
        # show image
        axs[0,0].imshow(img_rgb)
        axs[0,1].imshow(img_rgb)
        axs[1,0].imshow(img_rgb)
        axs[1,1].imshow(aff_mat, vmin=0, vmax=1)
        axs[1,1].set_title("Affinity matrix")
        
        # draw input video
        draw_bboxes(axs[0,0], 
                    bboxes, 
                    bboxes_type="xyxy", 
                    edge_color=[[1., 1., 0, 1]])
        axs[0,0].set_title("Frame: {: 6d} (Input)".format(frame_id))
        
        # draw bboxes for groups
        visualize_groups(axs[0,1], 
                        img_rgb//3*2, 
                        bboxes, 
                        node_clusters, 
                        node_group_type, 
                        fill_colors=group_color_dict)
        
        # draw queue end boxes
        draw_bboxes(axs[0,1], 
                    utils_fn.scale_bbox_xyxy(
                        bboxes[queue_end_idx], 
                        1.1), 
                    bboxes_type="xyxy", 
                    edge_color=[[1., 1., 0, 1]])
        
        axs[0,1].set_title('Result')
        
        
        # draw ground truth
        if plot_gt:
            visualize_groups(axs[1,0], 
                            img_rgb//3*2, 
                            bboxes, 
                            gt_node_clusters, 
                            gt_node_group_type, 
                            fill_colors=group_color_dict)
            
            if gt_queue_end_idx is not None and len(gt_queue_end_idx)>0:
                draw_bboxes(axs[1,0], 
                        utils_fn.scale_bbox_xyxy(
                            bboxes[gt_queue_end_idx], 
                            1.1), 
                        bboxes_type="xyxy", 
                        edge_color=[[1., 1., 0, 1]])
            
            axs[1,0].set_title('Ground truth')
        
        
        # save fig
        fig.savefig('./tmp/tmp0.jpg')
        imout = plt.imread('./tmp/tmp0.jpg')
        vid_writer.append_data(imout)
    
    vid_writer.close()
    os.remove('./tmp/tmp0.jpg')
