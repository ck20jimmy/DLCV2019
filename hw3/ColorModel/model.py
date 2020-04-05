import numpy as np
import sys
import os
from PIL import Image
import cv2

from tqdm import tqdm

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

from tensorboardX import SummaryWriter
from torch.nn.utils import spectral_norm


class ColorGenerator(nn.Module):


    def __init__(self):
        super(ColorGenerator, self).__init__()

        # 64x64
        self.FirstBlock = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 64x64
        self.Down1 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.Down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.Down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.Down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.Down5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2x2

        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.Up4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.Up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(32, 2, 1, 1, 0, bias=False),
            nn.Tanh()
        )


    def forward(self, inp_img):

        inp = self.FirstBlock(inp_img)  # 32

        down1 = self.Down1(inp)     #64
        down2 = self.Down2(down1)   #128
        down3 = self.Down3(down2)   #256
        down4 = self.Down4(down3)   #512
        down5 = self.Down5(down4)   #512

        up1 = self.Up1(down5)   #512

        up2_inp = torch.cat([up1, down4], dim=1)    #256
        up2 = self.Up2(up2_inp)

        up3_inp = torch.cat([up2, down3], dim=1)
        up3 = self.Up3(up3_inp)

        up4_inp = torch.cat([up3, down2], dim=1)
        up4 = self.Up4(up4_inp)

        up5_inp = torch.cat([up4, down1], dim=1)
        up5 = self.Up5(up5_inp)

        out = self.out(up5)

        return out



class ColorDiscriminator(nn.Module):

    def __init__(self):
        super(ColorDiscriminator, self).__init__()

        self.discriminate = nn.Sequential(

            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, img):
        return self.discriminate(img)
        

class ColorModel(nn.Module):

    def __init__(self):
        super(ColorModel, self).__init__()
        
        self.generator = ColorGenerator()
        self.discriminator = ColorDiscriminator()

    def generate(self, inp_img):
        return self.generator(inp_img)

    def discriminate(self, inp_img):
        return self.discriminator(inp_img)

