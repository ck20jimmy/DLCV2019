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
			nn.Dropout(0.5),
			nn.Linear(16384, 256),
			nn.SELU(inplace=True),
		)

		self.encoder = nn.GRU(256, 256, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)

		self.output_layer = nn.Sequential(
			nn.Linear(512, 256),
			nn.SELU(inplace=True),
			nn.Dropout(0.1),
			
			nn.Linear(256, 128),
			nn.SELU(inplace=True),
			nn.Dropout(0.1),

			nn.Linear(128, 11),
			nn.LogSoftmax(dim=1)
		)

	# (NxT, H, W)
	def extract(self, inp):
		with torch.no_grad():
			feat = self.Feature_Extract(inp).flatten(start_dim=1).detach()
		feat = self.RedDim(feat)
		return feat

	# (B, S, Emb Size)
	def encode(self, feat, inp_len):

		packed_emb = nn.utils.rnn.pack_padded_sequence(feat, inp_len, batch_first=True)
		output, hidden = self.encoder(packed_emb)

		cat_emb = hidden[-2:,:,:].permute(1,0,2).flatten(start_dim=1)
		out = self.output_layer(cat_emb)

		return out




