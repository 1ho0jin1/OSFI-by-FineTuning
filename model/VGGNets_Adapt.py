# VGGNet Implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
# Residual adapter implementation based on: https://github.com/srebuffi/residual_adapters

import torch.nn as nn

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
    
    

class VGG(nn.Module):
    def __init__(self, vgg_name, feat_dim=512, drop_ratio=0.5, adapt=False):
        super(VGG, self).__init__()
        self.adapt = adapt
        self.features = self._make_layers(cfg[vgg_name])
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          nn.Flatten(),
                                          nn.Linear(512 * 7 * 7, feat_dim),
                                          nn.BatchNorm1d(feat_dim))

    def forward(self, x):
        out = self.features(x)
        out = self.output_layer(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        
        if self.adapt:
            conv = conv_adapt
        else:
            conv = conv3x3
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(2)]
            else:
                layers += [conv(in_channels, x, bias=True),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}