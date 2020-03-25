# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:01:45 2020

@author: user
"""

import torch
from torch import nn

# https://github.com/amoudgl/pygoturn/blob/master/src/model.py
class New_GoTURN(torch.nn.Module):
    def __init__(self, model, inputshape = (1, 128, 128), inter_channels=4096, inter_conv_channels=256):
        super(New_GoTURN, self).__init__()
        
        # model.fc = nn.Linear(model.fc.in_features, model.fc.in_features)
        model.conv1= nn.Conv2d(inputshape[0], 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=True)
        # self.convnet = model
        bodylist = list(model.children())[:-2]
        self.convnet = torch.nn.ModuleList(bodylist)

        # disable any training except for batch norm in base model with resnext50
        train_prefix = ['bn']
        for name, param in self.convnet.named_parameters():
            if any(i in name for i in train_prefix):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.conv_adjuster = nn.Sequential(
            nn.Conv2d(model.fc.in_features, inter_conv_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_adjuster2 = nn.Sequential(
            nn.Conv2d(inter_conv_channels, inter_conv_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
#        self.regression_layer = nn.Sequential(
#            nn.Flatten(),
#            nn.Linear(256*5*5*2, inter_channels),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            nn.Linear(inter_channels, inter_channels),
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            # nn.Linear(4096, 4096),
#            # nn.ReLU(inplace=True),
#            # nn.Dropout(),
#            nn.Linear(inter_channels, 4)
#        )
        self.regression_layer = nn.Sequential(
            nn.Conv2d(inter_conv_channels*2, inter_conv_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(inter_conv_channels, inter_conv_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(inter_conv_channels, inter_conv_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AvgPool2d(5),
            nn.Flatten(),
            nn.Linear(inter_conv_channels, inter_conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(inter_conv_channels, 4),
            nn.ReLU(inplace=True)
        )
        self.weight_init()

    def weight_init(self):
        for m in self.regression_layer.modules():
            # fully connected layers are weight initialized with
            # mean=0 and std=0.005 (in tracker.prototxt) and
            # biases are set to 1
            # tracker.prototxt link: https://goo.gl/iHGKT5
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)
    
    def forward(self, x, y):
        # run first conv for base model (resnext50)
        for ii, fs in enumerate(self.convnet):
            x = fs(x)
            y = fs(y)
        x1 = x
        x2 = y
        # run second for mask inp
        x1 = self.conv_adjuster(x1)
        x1 = self.conv_adjuster2(x1)
        # print('x1 size: ', x1.size())
        # x1 = x1.view(x.size(0), -1)

        # run second for mask gt
        x2 = self.conv_adjuster(x2)
        x2 = self.conv_adjuster2(x2)
        # print('x2 size: ', x2.size())
        # x2 = x2.view(x2.size(0), -1)

        # combine both outputs, and put regression layers
        xcat = torch.cat((x1, x2), 1)
        # print('xcat shape: ', xcat.shape)
        x_out = self.regression_layer(xcat)
        return x_out