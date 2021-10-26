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
import cv2
import detectron2
import easyocr
from .utils_group_signs import bbox_xyxy2fourpoint, crop_align_nn, expand_bbox, dist_bboxes, merge_group
from skimage import measure
from collections import defaultdict
from scipy.sparse.csgraph import connected_components

class SignGroupConfigOption(object):
    """
    empty object for including attributes
    """
    pass

def detect_and_merge_results(img_rgb: np.array,
                             predictor: detectron2.engine.defaults.DefaultPredictor, 
                             reader_ocr: easyocr.easyocr.Reader, 
                             config_options: SignGroupConfigOption):
    """
    Detect objects using Detectron2 detector and EasyOCR reader then
    merge the result into a processable steps
    
    INPUT
    -------
    img_rgb     : np.array             - image to detector in RGB format
    predictor   : DefaultPredictor     - Detectron2 object detector
    reader_ocr  : Reader               - EasyOCR detector
    config_options : SignGroupConfigOption - config object
    
    OUTPUT
    -------
    dict_output : dictionary           - dictionary of output (see explanation of each item before return statement)
    """
    
    label_dict = config_options.label_dict

    # run object detector
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    outputs = predictor(img_bgr)

    # run ocr
    result_ocr = reader_ocr.readtext(img_rgb)


    ## merge the result of detector with the ocr results

    # read output from the object detector
    labels = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
    bboxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy()

    # merge the results
    bboxes_detectron = bboxes
    bboxes_4pt = np.array([bbox_xyxy2fourpoint(bbox) for bbox in bboxes])
    bboxes_4pt = bboxes_4pt[labels != label_dict['text']]
    labels = labels[labels != label_dict['text']]
    bboxes_ocr = np.array([res[0] for res in result_ocr])
    bboxes = np.concatenate([bboxes_4pt, bboxes_ocr], axis=0)
    labels = np.concatenate([labels, label_dict['text']*np.ones(len(bboxes_ocr), dtype=np.int)])

    dict_output = {}
    dict_output['bboxes'] = bboxes # bboxes in 4 point format, i.e., each point is represented as a 4 x 2 matrix
    dict_output['labels'] = labels # labels of each bbox
    dict_output['detector_output'] = outputs  # raw outout from the object detector
    dict_output['ocr_output'] = result_ocr  # raw output from the ocr reader

    return dict_output


def compute_laplacian_connected_components(img_rgb, config_options):
    """
    Compute connected components in input image based on its Laplacian image
    
    INPUT
    -------
    img_rgb        : n x m x 3 np.array     - input image in uint8 format
    config_options : SignGroupConfigOption  - config object
    
    OUTPUT
    -------
    img_cc         : n x m np.array         - image containing connected components ID
    num_cc         : int                    - number of the components
    """
    
    # get the params
    conncomps_img_scale = config_options.conncomps_img_scale
    conncomps_thres_same_color = config_options.conncomps_thres_same_color

    # scale image
    ims = cv2.resize(img_rgb, (0, 0), fx=conncomps_img_scale, fy=conncomps_img_scale)
    ims = cv2.cvtColor(ims, cv2.COLOR_BGR2RGB)

    imm = ims.copy()
    # imm = cv2.GaussianBlur(imm,(3,3),0)
    imm = cv2.Laplacian(imm, cv2.CV_16S)
    imm = np.abs(imm).max(axis=2)

    # find connected components

    imm2 = imm.copy()
    imm2 = np.abs(imm2) < conncomps_thres_same_color
    imm2 = imm2.astype(np.float32)
    img_cc, num_cc = measure.label(imm2, return_num=True)
    
    return img_cc, num_cc


def compute_group_signs(img_rgb: np.array, 
                        bboxes: np.array, 
                        labels: np.array, 
                        config_options: SignGroupConfigOption):
    """
    Group the detected objects into groups where each group contains
    one direction arrow and object (texts, etc.) belonging to the 
    directional arrow
    
    INPUT
    ------
    img_rgb        : n x m x 3 np.array     - image
    bboxes         : k x 4 x 2 np.array     - bboxes in 4-point format
    labels         : k np.array             - label 
    config_options : SignGroupConfigOption  - config object
    
    OUTPUT
    ------
    dict_output : dictionary           - dictionary of output (see explanation of each item before return statement)
    """
    
    # compute connected components
    img_cc, num_cc = compute_laplacian_connected_components(img_rgb, config_options)
    
    # get param
    conncomps_img_scale = config_options.conncomps_img_scale
    label_dict          = config_options.label_dict
    thres_min_ratio_cc  = config_options.groupsign_thres_min_ratio_cc
    param_expand_bbox   = config_options.groupsign_param_expand_bbox
    
    # crop and find the cc index for the crop
    bbox_cc = [None]*len(bboxes)
    for it_bbox, bbox in enumerate(bboxes):

        # for symbol expand the bbox a bit to account for square symbols 
        # (since the original square detection of square symbols may not include bg color)
        if labels[it_bbox] == label_dict['symbol']: 
            bbox_to_crop = expand_bbox(bbox, param_expand_bbox)*conncomps_img_scale
        else:
            bbox_to_crop = bbox*conncomps_img_scale

        # make sure the crop is inside the image
        bbox_to_crop[:, 0] = np.minimum(img_cc.shape[1]-1, np.maximum(0, bbox_to_crop[:, 0]))
        bbox_to_crop[:, 1] = np.minimum(img_cc.shape[0]-1, np.maximum(0, bbox_to_crop[:, 1]))

        # crop and find the cc index for the crop
        cropped_box = crop_align_nn(img_cc, bbox_to_crop)
        cc_label, cc_num = np.unique(cropped_box, return_counts=True)
        cc_num, cc_label = cc_num[cc_label != 0], cc_label[cc_label != 0]
        thres_tmp = np.size(cropped_box)*thres_min_ratio_cc
        cc_num, cc_label = cc_num[cc_num > thres_tmp], cc_label[cc_num > thres_tmp]
        bbox_cc[it_bbox] = cc_label

    # generate dictionary indicating connectivity (i.e., cc label to bbox id)
    dict_node = defaultdict(list)
    for it, it_bbox_cc in enumerate(bbox_cc):
        for num in it_bbox_cc:
            dict_node[num].append(it)

    # find bboxes sharing same cc label
    bbox_mat = np.zeros((len(bboxes), len(bboxes)))
    for it_key in dict_node:
        entry_tmp = dict_node[it_key]
        if len(entry_tmp) > 1:
            bbox_mat[entry_tmp[0], entry_tmp[1:]] = 1

    # # print cc and sign id correspondence
    # for it_key in dict_node:
    #     entry_tmp = dict_node[it_key]
    #     if len(entry_tmp) > 1:
    #         print(it_key, entry_tmp)

    # get signs in the same block
    num_asm, asm = connected_components(bbox_mat)


    output_block_labels = np.ones_like(asm, dtype=np.long)
    counter_group = 0
    for it_asm in range(num_asm):  

        # check nubmer of arrows in each block
        block_ids = np.where(asm==it_asm)[0]
        block_labels = labels[block_ids]
        num_arrows = sum(block_labels==label_dict['direction arrow'])
        # deal with different number of arrows

        if num_arrows <= 1:
            groups_tmp_cvt = [block_ids]

        elif num_arrows > 1:
            block_id_arrows = np.where(block_labels==label_dict['direction arrow'])[0]
            bbox_cvt = [None]*len(block_ids)
            for it_bbox, bbox_id in enumerate(block_ids):
    #             bbox_cvt[it_bbox] = bbox_xyxy2fourpoint(bboxes[bbox_id])
                bbox_cvt[it_bbox] = bboxes[bbox_id]

            # compute pairwise distance
            dist_mat = np.zeros((len(block_ids), len(block_ids)))
            for it1 in range(len(block_ids)):
                for it2 in range(it1+1, len(block_ids)):
                    val = dist_bboxes(bbox_cvt[it1], bbox_cvt[it2])
                    dist_mat[it2, it1] = val
                    dist_mat[it1, it2] = val

            # group the bbox based on arrows 
            merged_groups_tmp = merge_group(dist_mat, block_id_arrows)

            # convert back to original index
            groups_tmp_cvt = [[block_ids[it] for it in gr] for gr in merged_groups_tmp]

        # compute the output group
        for it_group in groups_tmp_cvt:
            output_block_labels[it_group] = counter_group
            counter_group += 1
    
    dict_output = {}
    dict_output['output_block_labels'] = output_block_labels # group index for each bbox
    dict_output['img_cc'] = img_cc   # image of connected components
    dict_output['num_cc'] = num_cc   # number of connected components
    return dict_output
