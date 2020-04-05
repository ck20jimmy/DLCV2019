import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys

import torchvision.models as models

class Model(nn.Module):

	def __init__(self, pretrain):
		super(Model, self).__init__()

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
		
		self.RedDim = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(16384, 256),
			nn.SELU(inplace=True),
		)

		self.encoder = nn.GRU(256, 256, batch_first=True, num_layers=1, bidirectional=True, dropout=0.2)

		self.output_layer = nn.Sequential(
			nn.Linear(512, 256),
			nn.SELU(inplace=True),
			nn.Dropout(0.1),

			nn.Linear(256, 11),
			nn.LogSoftmax(dim=1)
		)

	# (NxT, H, W)
	def extract(self, inp):
		with torch.no_grad():
			feat = self.Feature_Extract(inp).flatten(start_dim=1).detach()
		feat = self.RedDim(feat)
		return feat

	# (B, S, Emb Size)
	def encode(self, feat):
		output, hidden = self.encoder(feat)
		return output

	def classify(self, inp):
		out = self.output_layer(inp)
		return out


