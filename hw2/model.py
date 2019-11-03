import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import sys

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.conv1 = Conv2Times(in_channels, 64)
        
        self.Down = nn.ModuleList([
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
        ])
        
        self.Up = nn.ModuleList(
            Up(1024, 512),
            Up(512, 256),
            Up(256, 128),
            Up(128, 64)
        )
        
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, inp):
        
        down_elements = [ self.conv1(inp) ]
        
        for i in range(len(self.Down)):
            down_elements.append( self.Down[i](down_elements[i]) )
        
        up_element = self.Up(down_elements[-1], down_elements[-2])
        
        for i in range(3, len(self.Up)):
            up_element = self.Up(up_element, down_elements[-i])
        
        out = self.output_layer(up_element)
        
        return out

    
class Conv2Times(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Conv2Times, self).__init__()
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )
        
    def forward(self, inp):
        return self.conv(inp)
        
        
class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
    
        self.conv = Conv2Times(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, inp):
        return self.pool(self.conv(inp))
        
    
class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
    
        self.up_samp = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = Conv2Times(in_channels, out_channels)
        
    def forward(self, inp, prev_inp):
        
        crop_prev_inp = prev_inp
        
        up_inp = self.up_samp(inp)
        
        out = torch.cat([crop_prev_inp, up_inp], dim=1)
        out = self.conv(out)
        
        return out
        
class Up_special(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Up_special, self).__init__()
        self.conv = Conv2Times(in_channels*2, out_channels)
        
    def forward(self, inp, prev_inp):
        
        out = torch.cat([prev_inp, inp], dim=1)
        out = self.conv(out)
        
        return out


class ResUNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()
        
        self.resnet = models.resnet34(pretrained=True)
        
        self.Down1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        ) # 2
        
        self.Down2 = self.resnet.layer1 #2
        self.Down3 = self.resnet.layer2 #3
        self.Down4 = self.resnet.layer3 #4
        self.Down5 = self.resnet.layer4 #5
        
        self.Up1 = Up(512, 256)# 4
        self.Up2 = Up(256, 128)# 3
        self.Up3 = Up(128, 64)# 2
        
        self.Up3_5 = nn.Sequential(
            Conv2Times(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        )
        
        self.Up4_5 = Conv2Times(3, 16)
        
        self.Up4 = Up(64, 32)
        self.Up5 = Up(32, 16)
        
        self.output_layer = nn.Conv2d(16, out_channels, kernel_size=1)
        
    
    def forward(self, inp):
        
        d1 = self.Down1(inp)
        d2 = self.Down2(d1)
        d3 = self.Down3(d2)
        d4 = self.Down4(d3)
        d5 = self.Down5(d4)
        
        up_element = self.Up1(d5, d4)
        up_element = self.Up2(up_element, d3)
        up_element = self.Up3(up_element, d2)
        
        inp_35 = self.Up3_5(d1)
        up_element = self.Up4(up_element, inp_35)
        
        inp_45 = self.Up4_5(inp)
        up_element = self.Up5(up_element, inp_45)
        
        out = self.output_layer(up_element)
        
        return out
