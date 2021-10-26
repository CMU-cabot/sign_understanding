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

from torch.utils.data import Dataset
import torch
import numpy as np 
import os
from preprocessing import datadict2sample

import copy
from modules.utils import list_of_member_to_group_assignment
from shapely.geometry import Polygon
from skimage.draw import polygon2mask
import torchvision.transforms as transforms
import torch.nn.functional as F

class SignDataset(Dataset):
    
    def __init__(self, folder_location, n_categories, mode, preload=False, max_data=float('inf'), augment_crop=True):
        """
        folder_location : string       - location of the folder (will only read into immediate subfolders)
        n_categories    : int          - number fo categories 
        mode            : string       - 'train' or 'test' indicating whether to augment or not
        preload         : bool         - to preload data or not
        augment_crop    : bool         - augment data by random cropping
        """
        
        self.folder_location = folder_location
        self.preload = preload
        self.n_categories = n_categories
        self.mode = mode
        self.max_data = max_data
        self.augment_crop = augment_crop
        
        # list all files
        self.list_files = []
        self.samples    = []
        
        counter = 0
        for folder in os.listdir(folder_location):
            path_tmp = os.path.join(folder_location, folder)
            if not os.path.isdir(path_tmp):
                continue
            for filename in os.listdir(path_tmp):
                file_to_load = os.path.join(path_tmp, filename)
                if file_to_load[-4:] == '.npy':
                    self.list_files.append(file_to_load)
                    
                if preload:
                    datadict = np.load(file_to_load, allow_pickle=True).item()
#                     sample   = datadict2sample(datadict, n_categories)
                    samples.append(sample)
                
                counter += 1
        
        if max_data != float('inf'):
            self.list_files = self.list_files[:self.max_data]
        
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, idx):

        if self.preload:
            datasample = self.samples[idx]
        else:
            datadict = np.load(self.list_files[idx], allow_pickle=True).item()
        
        with torch.no_grad():
            datadict = random_crop_augment(datadict, to_augment=self.augment_crop)
            datasample = datadict2sample(datadict, self.n_categories)
            
            if self.mode == 'train':
                to_flip = np.random.rand() > 0.5
                if to_flip:
                    datasample.node_feature = torch.flip(datasample.node_feature, [3])
                    datasample.details = copy.deepcopy(datasample.details)
                    for i in range(len(datasample.details['segm'])):
                        datasample.details['segm'][i][:,0] = datasample.node_feature.shape[-1]-datasample.details['segm'][i][:,0]                    
        return datasample
    
    
def random_crop_augment(img_anno, to_augment=False, dim_min_size=256):
    """
    Randomly crop for augmentation
    Also edit other info, e.g., segm, cat_id, etc.
    
    INPUT
    -----
    img_anno     : dict    - annotated image with segm, cat_id, group information, etc.
    to_augment   : bool    - whether to run augment or not
    dim_min_size : int     - dimension of the minimum side
    
    OUTPUT
    ------
    datadict    : dict     - dictionary of the cropped data
    """
    
    # convert segmentation into polygon
    img  = img_anno['img_rgb_scaled']
    segm = copy.copy(img_anno['segm'])
    segm_poly = [Polygon(i) for i in segm]

    # randomize the crop frame for 10 times,
    # select the crop that have at least 2 segments,
    # otherwise just use the full image
    if to_augment:
        n_trials = 10
    else:
        n_trials = 0
    found = False
    s_y, s_x = img.shape[0], img.shape[1]  
    for it_try in range(n_trials):
        # generate the scale
        scale_r = 0.75+0.25*np.random.rand()
        scale_x = (0.8+0.2*np.random.rand())*scale_r
        scale_y = (0.8+0.2*np.random.rand())*scale_r

        # generate the offset 
        off_x = (1-scale_x)*np.random.rand()
        off_y = (1-scale_y)*np.random.rand()

        # compute the crop
        t, b, l, r = [int(i) for i in [off_y*s_y, (off_y+scale_y)*s_y, off_x*s_x, (off_x+scale_x)*s_x]]
        crop_img = np.array([[l,t],[l,b],[r,b],[r,t]])
        poly_img = Polygon(crop_img)

        # compute the number of intersections
        kept_segm_id = [] # kept the id of the segm still seen
        for it_poly, segm_poly_tmp in enumerate(segm_poly):
            segm_area = segm_poly_tmp.area
            if segm_area == 0:  # for some weird annotation with no area...
                segm_area=1
            iou_ratio = segm_poly_tmp.intersection(poly_img).area/segm_area
            if iou_ratio > 0.25:
                kept_segm_id.append(it_poly)

        if len(kept_segm_id) >= 2: # more than 2 segments left
            found=True
            break

    if found: # do the cropping and scaling
        segm[:,:,0] = segm[:,:,0]-off_x*s_x
        segm[:,:,1] = segm[:,:,1]-off_y*s_y
    else:
        kept_segm_id = np.arange(len(segm_poly))


    # generate mask
    masks = []
    for it_segm in img_anno['segm']:
        mask = polygon2mask(img.shape, it_segm[:,[1,0]]).astype(np.float)
        masks.append(mask[:,:,[0]])

    # concatenate
    node_feature_init = torch.from_numpy(np.concatenate([img]+masks, axis=2)).to(torch.float)
    node_feature_init = node_feature_init.permute([2,0,1])
    node_feature_crop = node_feature_init
    
    # scale image and segm
    if found:
        node_feature_crop = transforms.functional.crop(node_feature_init, t, l, b-t, r-l)

#         # reshape feature to specified size
    scale_tensor = dim_min_size/min(node_feature_crop.shape[1:])
    node_feature_crop = F.interpolate(node_feature_crop[None,], scale_factor=(scale_tensor,scale_tensor))[0]
    segm = segm*scale_tensor

    # keep only data of the segm that remains in the image
    datadict = {}
    datadict['cat_id'] = [img_anno['cat_id'][i] for i in kept_segm_id]
    datadict['segm'] = [segm[i] for i in kept_segm_id]
    datadict['node_group_id'] = [img_anno['node_group_id'][i] for i in kept_segm_id]
    dict_reorder = {j:i for i,j in enumerate(np.unique(datadict['node_group_id']))}
    datadict['node_group_id'] = np.array([dict_reorder[i] for i in datadict['node_group_id']])
    datadict['list_group_members'] = [np.where(img_anno['node_group_id']==i)[0] for i in range(len(dict_reorder))]
    datadict['mask'] = node_feature_crop[3:][kept_segm_id]
    datadict['img_rgb_scaled'] = node_feature_crop[:3]
    datadict['orig_segm'] = img_anno['orig_segm']
    datadict['filename']  = img_anno['filename']
    datadict['img_id']    = img_anno['img_id']
    
    return datadict
