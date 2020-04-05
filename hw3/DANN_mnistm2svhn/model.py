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

class GradientReversal(Function):

    @staticmethod
    def forward(ctx, inp, Lambda):
        ctx.Lambda = Lambda
        return inp.view_as(inp)

    @staticmethod
    def backward(ctx, grad):
        rev_grad = grad.neg() * ctx.Lambda
        return rev_grad, None



class DaNN(nn.Module):

    def __init__(self):
        super(DaNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.MaxPool2d(2),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def fit(self, inp_img, Lambda):
        feat = self.feature_extractor(inp_img)
        feat = feat.view(-1, 256 * 3 * 3)
        
        reverse_feat = GradientReversal.apply(feat, Lambda)

        class_out = self.class_classifier(feat)
        domain_out = self.domain_classifier(reverse_feat)

        return class_out, domain_out

    def predict(self, inp_img):
        feat = self.feature_extractor(inp_img)
        feat = feat.view(-1, 256 * 3 * 3)
        class_out = self.class_classifier(feat)

        return class_out
