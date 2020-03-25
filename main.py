# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:30:37 2020

@author: user
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T
import os

#custom dataset
from data_loader import ObjectDataset
from utils_rect import convert, \
                       get_rectange, \
                       draw_rectangle, \
                       widen_bbox, \
                       shift_mask, \
                       adjust_crop_image
#%matplotlib inline

def build_transform():#min_image_size=800):#(800, 1007)):
#    pixel_mean = [102.9801, 115.9465, 122.7717]
#    pixel_std = [1., 1., 1.]
#    normalize_transform = T.Normalize(
#        mean=pixel_mean, std=pixel_std
#    )

    transform = T.Compose(
        [
            T.ToPILImage(),
#            T.Resize(min_image_size),
#            T.ToTensor(),
#            to_bgr_transform,
#            normalize_transform,
        ]
    )
    return transform

# =============================================================================
# load image path
# =============================================================================
objpath = 'adjust_image'
objdestpath = 'D:\\ADJUST_MASK\\adjust_crop_image'
if 'obj' not in locals():
    obj  = ObjectDataset(objpath)
    imagelist, labellist = obj.load_image_and_gt(depth=False) 

# =============================================================================
# load  image transform
# =============================================================================
#imtransform = build_transform()

# =============================================================================
# visualize each highlighted image
# =============================================================================
highlighted = 2
highlighted_list = np.random.permutation(len(imagelist))
np.random.shuffle(highlighted_list)
highlighted_list = highlighted_list[:highlighted]
highlighted_list = [182, 52]
for i in range(len(imagelist[highlighted_list])):
    img = cv2.imread(imagelist[i])
    color = np.random.randint(0, 255, 3).tolist()
    
    #get masks of ith labels
    lbllist_i = labellist[i].copy()
    #get bboxes of ith labels
    lbllist_i_bbox = get_rectange(lbllist_i)
    # observe bbox
#    draw_rectangle(img, lbllist_i_bbox, color)
    # observe mask
#    cv2.drawContours(img, list(lbllist_i), -1, color, -1)
    
    # observe the shifted bbox
    lbllist_i_bbox_widen, label_percent = widen_bbox(percentage=0.8, 
                                          label_bbox=lbllist_i_bbox.copy(),
                                          imgsize=img.shape[:2][::-1], 
                                          mode='xyxy', 
                                          return_percent=True)
    draw_rectangle(img, lbllist_i_bbox_widen, color)
    
    # observe the shifted masks
    lbllist_i_shifted = shift_mask(label_percent, lbllist_i, img.shape[:2][::-1], random=True)
    cv2.drawContours(img, list(lbllist_i_shifted), -1, color, -1)
    
    # show the image output
    plt.imshow(img[...,::-1])
    plt.show()
    
    
    lbllist_i_bbox_shifted = get_rectange(lbllist_i_shifted)
    adjust_crop_image(objdestpath, 
                      imagelist[i].split('\\')[-1], 
                      img, 
                      lbllist_i_bbox_widen, 
                      lbllist_i_bbox, 
                      lbllist_i, 
                      lbllist_i_shifted)
    
    