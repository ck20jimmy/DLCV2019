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

from torch.autograd import Function




class Generater(nn.Module):

    def __init__(self):
        super(Generater, self).__init__()

        self.generate = nn.Sequential(
            nn.ConvTranspose2d(512+128+10+1, 64*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, inp, noise):

        inp = inp.view(-1, 128+10+1, 1, 1)
        noise = noise.view(-1, 512, 1, 1)

        gen_inp = torch.cat([inp, noise], dim=1)
        output = self.generate(gen_inp)

        return output



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminate = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64*2, 3, 1, 1),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64*2, 64*4, 3, 1, 1),           
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64*4, 64*2, 3, 1, 1),           
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(64*2, 10),
            nn.LogSoftmax(dim=1)
        )

        self.image_classifier = nn.Sequential(
            nn.Linear(64*2, 1),
            nn.Sigmoid()
        )

    def forward(self, inp_img):

        feat = self.discriminate(inp_img)

        feat = feat.view(-1, 64*2)

        class_prob = self.class_classifier(feat)
        image_prob = self.image_classifier(feat)

        return class_prob, image_prob



class FeatureExtracter(nn.Module):

    def __init__(self):
        super(FeatureExtracter, self).__init__()

        self.extract = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.MaxPool2d(2),

            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.MaxPool2d(2),

            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.MaxPool2d(2),

            # nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout2d(0.2),

            # nn.MaxPool2d(2),
            
            # Original
            nn.Conv2d(3, 64, 5, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 5, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 2 * 64, 4, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp_img):
        feat = self.extract(inp_img).view(-1, 2*64)
        return feat



class Source_Classifier(nn.Module):

    def __init__(self):
        super(Source_Classifier, self).__init__()

        self.classify = nn.Sequential(
            nn.Linear(2*64, 2*64),
            nn.ReLU(inplace=True),

            nn.Linear(2*64, 64),
            nn.ReLU(inplace=True),

            # nn.Linear(64, 64),
            # nn.ReLU(inplace=True),

            # nn.Linear(2*64, 10),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inp_img):
        return self.classify(inp_img)


class GTA(nn.Module):

    def __init__(self):
        super(GTA, self).__init__()

        self.feat_extracter = FeatureExtracter()
        self.source_classifier = Source_Classifier()
        self.generater = Generater()
        self.discriminator = Discriminator()
