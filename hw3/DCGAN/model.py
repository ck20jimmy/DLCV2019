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


class Generator(nn.Module):
    def __init__(self, gen_inp_size, image_size, image_channel):
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

    def forward(self, inp):
        return self.generate(inp)


class Discriminator(nn.Module):
    def __init__(self, inp_channel, image_size):
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

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.disciminate(inp)



class ContourGAN(nn.Module):

	def __init__(self, gen_inp_size, image_size, image_channel):
		super(ContourGAN, self).__init__()

		self.Generator = Generator(gen_inp_size, image_size, image_channel)
		self.Discriminator = Discriminator(image_channel, image_size)

	def generate(self, inp):
		out = self.Generator(inp)
		return out

	def discriminate(self, inp):
		out = self.Discriminator(inp)
		return out
