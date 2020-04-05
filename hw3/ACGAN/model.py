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

from torch.nn.utils import spectral_norm

from tensorboardX import SummaryWriter


from train import IMAGE_SIZE, INPUT_CHANNEL, NOISE_SIZE


class Generator(nn.Module):
    def __init__(self, gen_inp_size=128, image_size=64, image_channel=1):
        super(Generator, self).__init__()
        
        self.generate = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(gen_inp_size, 512, 4, 1, 0, bias=False ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, image_channel, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, inp, class_inp):
        gen_inp = torch.cat([inp, class_inp], dim=1).view(-1, 128, 1, 1)
        return self.generate(gen_inp)


class Discriminator(nn.Module):
    def __init__(self, inp_channel=1, image_size=64):
        super(Discriminator, self).__init__()
        
        self.disciminate = nn.Sequential(

            nn.Conv2d(inp_channel, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.real_fake_layer = nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        self.smile_layer = nn.Conv2d(256, 1, 4, 1, 0, bias=False)

        self.real_fake_module = nn.Sequential(
            self.real_fake_layer,
            nn.Sigmoid()
        )

        self.smile_module = nn.Sequential(
            self.smile_layer,
            nn.Sigmoid()
        )


    def forward(self, inp):

        feat = self.disciminate(inp)

        real_fake_pred = self.real_fake_module(feat)
        smile_pred = self.smile_module(feat)

        return real_fake_pred, smile_pred


class ACGAN(nn.Module):

    def __init__(self):
        super(ACGAN, self).__init__()

        self.Generator = Generator(NOISE_SIZE)
        self.Discriminator = Discriminator(INPUT_CHANNEL)

    def generate(self, inp, class_inp):
        return self.Generator(inp, class_inp)

    def discriminate(self, inp):
        return self.Discriminator(inp)



