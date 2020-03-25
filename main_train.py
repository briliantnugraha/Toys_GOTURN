# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:32:00 2020

@author: user
"""

print('0. load libraries...')
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import models
from time import time
import matplotlib.pyplot as plt
import cv2

# custom libraries
from main_aug import random_input
from goturn import New_GoTURN

class AdjustMask:
    def __init__(self,
                inputshape = (1, 128, 128),
                out_features = 4,
                device = torch.device('cuda'),
                batchsize = 16,
                step = 10000,
                lr = 1e-4,
                losslog = 10):
        print('1. load HYPERPARAMETERS...')
        self.inputshape = inputshape
        self.out_features = out_features # to predict x1y1 and x2y2
        self.device = device
        self.batchsize = batchsize
        self.step = step
        self.lr = lr
        self.losslog = losslog
        self.base_model = self.load_base_model()
        self.model_custom = self.load_custom_model()
        
    def load_base_model(self):
        print('2. load Base Model...')
        model = models.resnext50_32x4d(pretrained=True)
        model.to(device)
        model.eval()
        model.conv1= nn.Conv2d(self.inputshape[0], 64, 
                               kernel_size=(3, 3), 
                               stride=(2, 2), 
                               padding=(3, 3), 
                               bias=True)
        return model
    
    def load_custom_model(self):
        print('3. load Custom Model...')
        model_custom = New_GoTURN(self.base_model).to(device)
        return model_custom
    
    def check_inference_time(self, rtime=5):
        print('...{}x Inference-test of model_custom'.format(rtime,inputshape))
        dummy_inp = torch.rand((1,)+inputshape).to(device)
        with torch.no_grad():
            for i in range(rtime):
                start = time()
                _ = self.model_custom(dummy_inp, dummy_inp)
                end = time()
                print('[Test-{}] Runtime: {:.3f}s'.format(i+1, end-start))
                
    def prepare_training_materials(self):
        # prepare training materials
        print('5. load Model Loss and Optimizers...')
        self.lossfunc = torch.nn.SmoothL1Loss(reduction='sum')
        #lossfunc = torch.nn.MSELoss(reduction='sum')
        # optimizer = optim.SGD(model.parameters(), lr=1e-5)
        self.optimizer = optim.Adam(self.model_custom.parameters(), lr=self.lr)
        self.running_loss = 0.0
        self.running_smallest = 1e+8
        self.running_loss_log = []
    
    def load_weight(self, name_smallest='trained_resnext50_32x4d.pth'):
        print('7. Load trained model...')
        self.model_custom.load_state_dict(
                        torch.load('trained_resnext50_32x4d.pth')
                        )
        
    def train(self, 
               save_smallest=True, 
               name_smallest='trained_resnext50_32x4d.pth'):
        self.prepare_training_materials()
        self.model_custom.train()
        print('6. Start Training...')
        for i in range(self.step):
            start = time()
            # reset gradient
            self.optimizer.zero_grad()
            # load data and labels
            mask_gt, mask_inp, gt_rect = random_input(self.device,
                                                 self.inputshape[1:], 
                                                 self.batchsize, 
                                                 savedata=False, 
                                                 debug=-1, 
                                                 progress=False)
            minput_combined1 = mask_gt[:, :1]
            minput_combined2 = mask_inp[:, :1]
            
            # forward pass
            outputs = self.model_custom(minput_combined1, minput_combined2)
            # compute loss function
            loss= self.lossfunc(outputs, gt_rect)
            # compute the gradient for the loss
            loss.backward()
            # do gradient descent
            self.optimizer.step()
            end = time()
            
            # print statistics
            get_loss = loss.item()
            self.running_loss += get_loss
            if (i+1) % losslog == 0:    # print every 2000 mini-batches
                print('[%5d/%d] loss: %.5f, smallest loss: %.5f'
                      ', time/step: %.3f s' % (i + 1,
                                              self.step,
                                              self.running_loss / self.losslog, 
                                              self.running_smallest,   
                                              end-start))
                self.running_loss_log.append(self.running_loss)
                self.running_loss = 0.0
            if self.running_smallest > get_loss:
                self.running_smallest = get_loss
                if save_smallest:
                    torch.save(self.model_custom.state_dict(), name_smallest)
        
    def detect(self, x1, x2, gt_rect):
        print('8. Check inference...')
        self.model_custom.eval()
        with torch.no_grad():
            outputs = self.model_custom(x1, x2).cpu().numpy()
            gt_rect_out = gt_rect.cpu().numpy()
            return outputs, gt_rect_out
        
    def produce_dummy_input(self, inp_custom=None, batchsize=None):
        inp_custom = self.inputshape[1:] if inp_custom is None else inp_custom[1:]
        batchsize    = self.batchsize if batchsize is None else batchsize
        mask_gt, mask_inp, gt_rect = random_input(self.device,
                                                 inp_custom, 
                                                 batchsize, 
                                                 savedata=False, 
                                                 debug=-1, 
                                                 progress=False)
        minput_combined1 = mask_gt[:, :1]
        minput_combined2 = mask_inp[:, :1]
        return minput_combined1, minput_combined2, gt_rect

if __name__ == '__main__':
    # define hyperparameters
    inputshape = (1, 128, 128)
    out_features = 4
    device = torch.device('cuda')
    batchsize = 16
    step = 10000
    losslog = 10
    lr = 1e-4
    
    # call the object
    adjust_mask = AdjustMask(inputshape = inputshape,
                            out_features = out_features,
                            device = device,
                            batchsize = batchsize,
                            step = step,
                            lr = lr,
                            losslog = losslog)
    # load trained weight if exists
    adjust_mask.load_weight()
    # do training
#    adjust_mask.train()
    
    # test detect with dummy input
    x1, x2, gt_rect = adjust_mask.produce_dummy_input(inputshape, batchsize)
    outputs, gt_rect_out = adjust_mask.detect(x1, x2, gt_rect)
    
    result2 = x1.cpu().numpy().copy()#.astype(np.uint8)
    gt_rect_out_np = np.array([(gro * np.repeat(inputshape[1:],2,0)).astype(int) \
                            for gro in gt_rect_out])
    for i in range(len(outputs)//4):
        print('N-{}: {} vs {}'.format(i+1, outputs[i]* np.repeat(inputshape[1:],2,0), gt_rect_out_np[i]))
        cv2.circle(result2[i,0], tuple(gt_rect_out_np[i,:2]), 3, 0.5, 3) 
        cv2.circle(result2[i,0], tuple(gt_rect_out_np[i,2:]), 3, 0.5, 3) 
        plt.imshow(result2[i,0])
        plt.show()
        
    