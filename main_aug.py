# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:13:14 2020

@author: user
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T
import os
from time import time
import torch
from PIL import Image


def build_transform(min_size=(128,128)):

    to_bgr_transform = T.Lambda(lambda x: x / 255)
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_size),
            T.ToTensor(),
#            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#            to_bgr_transform,
        ]
    )
    return transform

def random_input(device, imgsize, total_input_data, savedata=False, debug=-1, progress=False):
    mask_cs_list = []
    mask_depth_list = []
    mask_cs_rect_list = []
    if progress: start = time()
    for i in range(total_input_data):
        # make dummy depth image
        size = [np.random.uniform(0.5, 1.0) * arr for arr in (256, 256)]
        size = np.array(size).astype(int)
        mask_cs    = np.zeros(size)
        mask_depth = np.zeros(size)
        
        # randomized the rectangle of image gt
        rand_rect1=np.random.uniform(0.0, 0.4)
        rand_rect2=np.random.uniform(rand_rect1+0.25, np.maximum(1.0, rand_rect1+0.75) )
        mask_cs_rect_xy1 = (rand_rect1 * np.array(mask_cs.shape[::-1])).astype(int)
        mask_cs_rect_xy2 = (rand_rect2 * np.array(mask_cs.shape[::-1])).astype(int)
        # limit the input 0 to size -1
        mask_cs_rect_xy1 = np.maximum(0, np.minimum(size[::-1]-1, mask_cs_rect_xy1))
        mask_cs_rect_xy2 = np.maximum(0, np.minimum(size[::-1]-1, mask_cs_rect_xy2))
        # stack xy1 and xy2 together
        mask_cs_rects = np.hstack((mask_cs_rect_xy1,mask_cs_rect_xy2, size[::-1]))
        
        # draw random rectangle of rects and re adjust axis
        cv2.rectangle(mask_cs, tuple(mask_cs_rect_xy1), tuple(mask_cs_rect_xy2), 255, -1)
        mask_cs = np.repeat(mask_cs[None,...], 3, 0).astype(np.uint8)
        mask_cs = np.moveaxis(mask_cs, 0, -1)
        
        # set  image input to the center
        mask_depth_rect_xy1 = (size /4).astype(int)[::-1]
        mask_depth_rect_xy2 = size[::-1] - mask_depth_rect_xy1
        
        # draw center rectangle and re adjust axis
        cv2.rectangle(mask_depth, tuple(mask_depth_rect_xy1), tuple(mask_depth_rect_xy2), 255, -1)
        mask_depth = np.repeat(mask_depth[None,...], 3, 0).astype(np.uint8)
        mask_depth = np.moveaxis(mask_depth, 0, -1)
        
        # transform into pytorch
#        print(np.min(mask_cs), np.max(mask_cs), mask_cs.shape)
        mask_inp, mask_gt, gt_rect = transform_pytorch(device, imgsize, mask_cs, mask_depth, mask_cs_rects)
        
        # append the images
#        mask_cs_list.append(mask_cs)
#        mask_depth_list.append(mask_depth)
        mask_cs_list.append(mask_gt)
        mask_depth_list.append(mask_inp)
        mask_cs_rect_list.append(gt_rect)
        # save the data
        if savedata:
            cv2.imwrite(os.path.join(basepath, 'img{}_gt.png'.format(str(i))), mask_cs)
            cv2.imwrite(os.path.join(basepath, 'img{}_input.png'.format(str(i))), mask_depth)
        if i < debug:
            # show the image output
            gt_rect_out = gt_rect.cpu().numpy()
            gt_rect_out[:2] = (gt_rect_out[:2]*np.array(imgsize[::-1])).astype(int)
            gt_rect_out[2:] = (gt_rect_out[2:]*np.array(imgsize[::-1])).astype(int)
            print(gt_rect_out)
            mc_draw = mask_gt[0,...].cpu().numpy().copy() *255
            md_draw = mask_inp[0,...].cpu().numpy().copy()
            print('mc_draw: ', mc_draw.shape, ', md_draw: ', md_draw.shape)
            cv2.circle(mc_draw, tuple(gt_rect_out[:2]), 3, 128, 3) 
            cv2.circle(mc_draw, tuple(gt_rect_out[2:]), 3, 128, 3) 
            # show
            plt.subplot(121)
            plt.imshow(mc_draw)
            plt.subplot(122)
            plt.imshow(md_draw)
            plt.show()
        if progress:
            print('{}-done... {}, {}, {}, {}'.format(i, mask_inp.shape, 
                      mask_inp.mean(), mask_gt.shape, mask_gt.mean()))
    if progress: 
        end = time()
        print('Image generation time: {:.3f}s'.format(end-start))
    mask_cs_list      = torch.stack(mask_cs_list)
    mask_depth_list   = torch.stack( mask_depth_list)
    mask_cs_rect_list = torch.stack( mask_cs_rect_list)
    return mask_cs_list, mask_depth_list, mask_cs_rect_list

def transform_pytorch(device, imgsize, mask_cs_list, mask_depth_list, mask_cs_rect_list):
    '''
    imgsize format (h, w)
    mask_cs_list = list of mask gt (could contain different sizes)
    mask_depth_list = list of mask input (could contain different sizes)
    mask_cs_rect_list = array of rect gt (N, 6) with format  x1, y1, x2, y2, w,h
    '''
    imtransform = build_transform(imgsize[::-1])
    mask_gt  = imtransform(mask_cs_list).to(device)
    mask_inp = imtransform(mask_depth_list).to(device)
#    gt_rect_multiplier = np.array(imgsize[::-1]) / mask_cs_rect_list[4:][::-1]
#    gt_rect_multiplier = np.hstack((gt_rect_multiplier, gt_rect_multiplier)).astype(int)
    gt_rect_multiplier = torch.Tensor(mask_cs_rect_list[4:])
    gt_rect_multiplier = torch.stack((gt_rect_multiplier, gt_rect_multiplier)).flatten()#.type(torch.int)
    gt_rect = torch.div(torch.Tensor(mask_cs_rect_list[:4]), gt_rect_multiplier)
    gt_rect = torch.Tensor(gt_rect).to(device)
    return mask_inp, mask_gt, gt_rect
    
if __name__ == '__main__':
    # HYPERPARAMETERS
    device = torch.device('cuda')
    basepath = 'adjust_crop_image'
    min_image_size=(128,128)
    total_input_data = int( input('Input your total data (int): ') )
    mask_gt, mask_inp, gt_rect = random_input(device,
                                             min_image_size, 
                                             total_input_data, 
                                             savedata=False, 
                                             debug=False, 
                                             progress=False)

