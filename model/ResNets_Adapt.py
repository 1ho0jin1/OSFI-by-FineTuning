"""
@author: Jun Wang    
@date: 20201019   
@contact: jun21wangustc@gmail.com 
"""

# based on:  
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

# Residual adapter implementation based on: https://github.com/srebuffi/residual_adapters

import torch.nn as nn
from collections import namedtuple


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class conv_adapt(nn.Module):
    def __init__(self, in_planes, planes, stride=1, bias=False):
        super(conv_adapt, self).__init__()
        self.conv3x3 = conv3x3(in_planes, planes, stride, bias)
        self.conv1x1 = conv1x1(in_planes, planes, stride)
        nn.init.constant_(self.conv1x1.weight ,0)  # initialize 1x1 weights as 0s
    
    def forward(self, x):
        y = self.conv3x3(x)
        y += self.conv1x1(x)
        return y

    
class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut
    
class bottleneck_IR_adapt(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_adapt, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            conv_adapt(in_channel, depth, stride=1, bias=False), nn.PReLU(depth),
            conv_adapt(depth, depth, stride, bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

    
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

#class Backbone(Module):
class Resnet(nn.Module):
    def __init__(self, num_layers, drop_ratio, feat_dim=512, out_h=7, out_w=7, adapt=True):
        super(Resnet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        blocks = get_blocks(num_layers)
        if adapt:
            unit_module = bottleneck_IR_adapt
        else:
            unit_module = bottleneck_IR
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      nn.BatchNorm2d(64), 
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), 
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(512 * out_h * out_w, feat_dim), # for eye
                                       nn.BatchNorm1d(feat_dim))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x
