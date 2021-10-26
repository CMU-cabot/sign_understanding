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

def crop_image(img, bbox):
    # bbox format [x_min, y_min, x_max, y_max]
    
    # round
    bbox = np.round(bbox).astype(np.int)

    if len(img.shape)==2:
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    
def expand_bbox(bboxes, p):
    """
    expand size of bbox by p percentage
    bbox format: 4 pts
    """
    
    if len(bboxes.shape) == 2:
        one_dim = True
        bboxes = bboxes[None, :]
    else:
        one_dim = False
    
    bbox_max = np.max(bboxes, axis=1, keepdims=True)
    bbox_min = np.min(bboxes, axis=1, keepdims=True)
    
    bbox_c = (bbox_max+bbox_min)/2
    

    bbox_disp = bboxes - bbox_c
    bboxes_out = bbox_disp*(1+p/2) + bbox_c
    
    if one_dim:
        return bboxes_out[0]
    
    return bboxes_out


def dist_pt2lineseg(pt, pt1, pt2):
    """
    compute the distance from point pt to 
    a line segment defined by pt1 and pt2
    """
    pt1 = pt1.squeeze()[None,:]
    pt2 = pt2.squeeze()[None,:]
    
    displacement = (pt1-pt2)
    norm_sq = np.sum(displacement**2)
    alpha = np.sum((pt-pt2) * displacement, axis=1)/norm_sq
    
    alpha = np.maximum(0, np.minimum(1, alpha))
    alpha = alpha[:, None]
    proj_pt = alpha*pt1 + (1-alpha)*pt2
    
    dist = np.linalg.norm(pt - proj_pt, axis=1)
    
    return dist


def dist_lineseg(p11, p12, p21, p22):
    """
    Find the optimal transport distance between 2 line segments
    defined each by 2 endpoints
    """
    
    alpha = np.linspace(0, 1, 10)[:, None]
    p11 = p11.squeeze()[None,:]
    p12 = p12.squeeze()[None,:]
    p21 = p21.squeeze()[None,:]
    p22 = p22.squeeze()[None,:]
    
    v1 = np.mean(dist_pt2lineseg(alpha*p11+(1-alpha)*p11, p21, p22))
    v2 = np.mean(dist_pt2lineseg(alpha*p21+(1-alpha)*p21, p11, p12))
    
    return min(v1, v2)


def bbox_xyxy2fourpoint(bbox):
    """
    Convert bbox from xyxy format to 4 point format
    """
    
    return np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],  
        [bbox[0], bbox[3]],
    ])


def dist_bboxes(box1, box2):
    """
    Find the distance between two boxes (xyxy)
    defined as the (approx.) minimum distance to transport 
    one side of a box to one side of the other box
    
    NOTE THAT THIS IS NOT A GOOD DISTANCE!!
    since if one side of a box is completely inside another box
    then it the distnace should be 0, but this function will
    not give such distance...
    but it is approximately ok...
    
    In anycasem we can define a new distance and replace
    it here in the future.
    
    Input
    ------
    box1   : 4 np.array       - box 1
    box2   : 4 np.array       - box 2
    
    Output
    ------
    dist   : float            - distance betweem two boxes
    """
    
    min_dist = float('inf')
    
    for side1 in range(4):
        for side2 in range(4):
            dist_tmp = dist_lineseg(box1[side1], box1[(side1+1) % 4], box2[side2], box2[(side2+1) % 4])
            min_dist = min(min_dist, dist_tmp)
    return min_dist


def merge_group(dist_mat, head_id):
    """
    Merge the data into groups based on distance
    and head_id (kind of like constrained clustering)
    """
    
    # initialization
    avail = set(np.arange(len(dist_mat)))
    output = []
    dist_list = []
    for it in head_id:
        avail.remove(it)
        output.append(set([it])) # perpare output
        dist_list.append(dist_mat[it].copy())
    dist_list = np.array(dist_list)
    dist_list[:, head_id] = float('inf')
    
    id_range = np.arange(len(head_id))
    while avail:
        
        # find the minimum distance in dist_list
        amin = np.argmin(dist_list, axis=1)
        min_val = dist_list[id_range, amin]
        min_row = np.argmin(min_val)
        min_col = amin[min_row]
        
        # replace the connection to the min_col node
        dist_list[:, min_col] = float('inf')
        
        # save the new min
        output[min_row].add(min_col)
        avail.remove(min_col)
        
        # update the distance
        list_avail = list(avail)
        dist_list[min_row, list_avail] = np.minimum(dist_list[min_row, list_avail], dist_mat[min_col, list_avail])
        
    output = [list(it_output) for it_output in output]
        
    return output


def crop_align_nn(im_nn, bbox):
    """
    Crop image based on the bbox and warp it with nearest neighbor
    """
    
    grid_size = (50, 50)
    Xg = np.linspace(0, 1, grid_size[0])
    Yg = np.linspace(0, 1, grid_size[1])

    x1 = (1-Xg)*bbox[0][0] + (Xg)*bbox[1][0]
    x2 = (1-Xg)*bbox[3][0] + (Xg)*bbox[2][0]
    X = x1[None,:] * (1-Yg[:,None]) + x2[None,:] * (Yg[:,None])

    y1 = (1-Yg)*bbox[0][1] + (Yg)*bbox[3][1]
    y2 = (1-Yg)*bbox[1][1] + (Yg)*bbox[2][1]
    Y = y1[:,None] * (1-Xg[None,:]) + y2[:,None] * (Xg[None,:])

    X = np.round(X).astype(np.int)
    Y = np.round(Y).astype(np.int)

    im_out = im_nn[Y, X]

    return im_out
