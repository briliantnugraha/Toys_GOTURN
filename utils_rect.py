# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:56:48 2020

@author: user
"""
import cv2
import numpy as np
import os

def convert(bbox, mode_src='xyxy', mode_dst='xywh'):
    if mode_src=='xyxy' and mode_dst =='xywh':
        x, y = (bbox[:2] + bbox[2:])//2
        w, h = bbox[2:] - bbox[:2]
        bbox = np.array([x, y, w, h])
    elif mode_src=='xywh' and mode_dst =='xyxy':
        x1, y1 = bbox[:2] - (bbox[2:]//2)
        x2, y2 = bbox[:2] + (bbox[2:]//2)
        bbox = np.array([x1, y1, x2, y2])    
    return bbox

def get_rectange(label_mask):
    label_bbox = []
    for i in range(len(label_mask)):
        x1y1 = np.min(label_mask[i], axis=0)
        x2y2 = np.max(label_mask[i], axis=0)
        x1y1x2y2 = np.hstack((x1y1, x2y2))
        label_bbox.append(x1y1x2y2)
    return np.array(label_bbox)


def draw_rectangle(img, bbox, color=(255,0,0)):
    color=[255-c if c < 128 else c for c in color]
    for i in range(len(bbox)):
        cv2.rectangle(img, tuple(bbox[i,:2]), tuple(bbox[i,2:]), color, 3)

def widen_bbox(percentage=0.1, label_bbox=None, imgsize=None, mode='xyxy', return_percent=False):
    '''
    percentage: percentage of widen the bbox 0 to 1 or [0,1]
    label_bbox: the bbox to widen
    imgsize: tuple(, h)
    mode: the base mode
    '''
    new_bbox = []
    new_percent = []
    for i in range(len(label_bbox)):
        # convert to xywh for easier widening
        bbox_xywh = convert(label_bbox[i], mode_src=mode, mode_dst='xywh')
        percent = (percentage * np.array(bbox_xywh[2:])).astype(int)
        bbox_xywh[2:] += percent
        
        # convert to xyxy back to make sure it is 0 <= x <= imgsize
        bbox_xyxy = convert(bbox_xywh, mode_src='xywh', mode_dst=mode)
        bbox_xyxy = np.maximum(0, np.minimum(bbox_xyxy, np.array(imgsize*2)-1 ))
#        print('bbox: ', label_bbox[i], bbox_xyxy)
        
        new_bbox.append(bbox_xyxy)
        new_percent.append(percent)
    if return_percent:
        return np.array(new_bbox), np.array(new_percent)
    else :
        return np.array(new_bbox)

def shift_mask(label_percent, label_mask, imgsize, random=False):
    label_percent = label_percent //2
    
    label_outmask = []
    for i in range(len(label_mask)):
        label_percent_prob = (np.random.randn()*label_percent[i]).astype(int) if random else label_percent[i]
#        label_m  = label_mask[i].copy() - label_percent[i]
        label_m  = label_mask[i].copy() - label_percent_prob
        label_m  = np.maximum(0, np.minimum(label_m, np.array(imgsize)-1)).astype(int)
        _, index = np.unique(label_m, axis=0, return_index=True)
        index    = np.sort(index)
        label_m  = label_m[index]
        label_outmask.append(label_m)
#        print(label_mask[i].shape, 
#              label_percent[i].shape, 
#              np.min(label_mask[i], axis=0), 
#              np.max(label_mask[i], axis=0))
    return label_outmask

def adjust_crop_image(path_out, 
                      imgname, 
                      img, 
                      lbllist_i_bbox_widen, 
                      lbllist_i_bbox, 
                      lbllist_i, 
                      lbllist_i_shifted):
    print('path_out: ', path_out)
    lbllist_i_bbox_shifted = get_rectange(lbllist_i_shifted)
    lbllist_i_bbox_shifted_widen = widen_bbox(percentage=0.8, 
                                  label_bbox=lbllist_i_bbox_shifted.copy(),
                                  imgsize=img.shape[:2][::-1], 
                                  mode='xyxy', 
                                  return_percent=False)
    imgname_split = imgname.split('.')
    for i in range(len(lbllist_i_bbox)):
        w, h = lbllist_i_bbox_shifted_widen[i][2:] - lbllist_i_bbox_shifted_widen[i][:2]
        
        
        # draw for input
        mask_inp = np.zeros((h, w)).astype(np.int)
        mask_lbl_inp = lbllist_i_shifted[i] - lbllist_i_bbox_shifted_widen[i][:2]
        mask_lbl_inp  = np.maximum(0, np.minimum(mask_lbl_inp, np.array(mask_inp.shape[::-1])-1)).astype(int)
        _, index = np.unique(mask_lbl_inp, axis=0, return_index=True)
        index    = np.sort(index)
        mask_lbl_inp  = mask_lbl_inp[index].reshape((1, -1, 2))
        
        # draw the contour and save the image
        cv2.drawContours(mask_inp, [mask_lbl_inp.astype(int)], -1, 255, -1)
        stat1 = cv2.imwrite(os.path.join(os.getcwd(), path_out, imgname_split[0]+'_'+str(i)+'_input.'+imgname_split[1]), mask_inp.astype(np.uint8))
        
        # draw for ground-truth
        mask_gt = np.zeros((h, w)).astype(np.int)
        mask_lbl_gt = lbllist_i[i] - lbllist_i_bbox_shifted_widen[i][:2]
        mask_lbl_gt  = np.maximum(0, np.minimum(mask_lbl_gt, np.array(mask_gt.shape[::-1])-1)).astype(int)
        _, index = np.unique(mask_lbl_gt, axis=0, return_index=True)
        index    = np.sort(index)
        mask_lbl_gt  = mask_lbl_gt[index].reshape((1, -1, 2))
        
        # draw the contour and save the image
        cv2.drawContours(mask_gt, list(mask_lbl_gt), 0, 255, -1)
        stat2 = cv2.imwrite(os.path.join(path_out, imgname_split[0]+'_'+str(i)+'_gt.'+imgname_split[1]), mask_gt.astype(np.uint8))
        newpath= os.path.join(path_out, imgname_split[0]+'_'+str(i)+'_gt.'+imgname_split[1])
        print('Status (1,2): {}, {}, path: {}'.format(stat1, stat2, newpath))
        