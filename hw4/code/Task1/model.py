import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys

import torchvision.models as models

class Model(nn.Module):

	def __init__(self, pretrain):
		super(Model, self).__init__()

		# self.pretrain = models.resnet50(pretrained=True)
		self.Feature_Extract = nn.Sequential(
			pretrain.conv1,
			pretrain.bn1,
			pretrain.relu,
			pretrain.maxpool,

			pretrain.layer1,
			pretrain.layer2,
			pretrain.layer3,
			pretrain.layer4,

			pretrain.avgpool
		)
		
		self.classifier = nn.Sequential(

			nn.Linear(16384, 16384//4),
			nn.SELU(inplace=True),
			nn.Dropout(0.5),

			nn.Linear(16384//4, 16384//16),
			nn.SELU(inplace=True),
			nn.Dropout(0.5),

			nn.Linear(16384//16, 16384//64),
			nn.SELU(inplace=True),
			nn.Dropout(0.2),

			nn.Linear(16384//64, 16384//128),
			nn.SELU(inplace=True),
			nn.Dropout(0.2),

			nn.Linear(16384//128, 11),
			nn.LogSoftmax(dim=1)
		)

	# (NxT, H, W)
	def extract(self, inp):
		with torch.no_grad():
			feat = self.Feature_Extract(inp).flatten(start_dim=1).detach()
		return feat

	# (N, Feature)
	def classify(self, inp_feat):
		prob = self.classifier(inp_feat)
		return prob