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

from data import ImageDataset
from model import *
from train import *


# https://arxiv.org/pdf/1704.00028.pdf
# https://arxiv.org/pdf/1409.1556v6.pdf

if __name__ == '__main__':

	# transform = trns.Compose([ trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
	transform = trns.Compose([])

	mnist_data = ImageDataset("../../hw3_data/digits/mnistm/train/", \
		"../../hw3_data/digits/mnistm/train.csv", \
		transform )
	mnist_dataloader = DataLoader(mnist_data, batch_size=128, shuffle=True, num_workers=1)

	svhn_data = ImageDataset("../../hw3_data/digits/svhn/train/", \
		"../../hw3_data/digits/svhn/train.csv", \
		transform )
	svhn_dataloader = DataLoader(svhn_data, batch_size=128, shuffle=True, num_workers=1)

	mnistm_testdata = ImageDataset("../../hw3_data/digits/mnistm/test/", \
		"../../hw3_data/digits/mnistm/test.csv", \
		transform )
	mnistm_test_dataloader = DataLoader(mnistm_testdata, batch_size=128, shuffle=False, num_workers=2)

	model = DaNN()

	learning_rate = 1e-3
	# model_optim = torch.optim.SGD(model.parameters(), lr=learning_rate,  momentum=0.9)
	model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)


	alpha = 10
	beta = 0.75

	epochs = 120

	save_dir = "../../model/DaNN_Svhn2Mnistm/MnistM/"

	max_accuracy = -1


	model = model.cuda()
	
	for ep in range(epochs):

		print("Epoch {}".format(ep+1))

		p = (ep) / epochs
		# for param_group in model_optim.param_groups:
			# param_group['lr'] = learning_rate / np.power((1 + alpha * p), beta)

		Lambda = torch.FloatTensor([2 / (1+np.exp(-10 * p)) - 1]).cuda()

		print("Training ...")
		# model_train(model, model_optim, svhn_dataloader ,mnist_dataloader , Lambda)
		model_train(model, model_optim, mnist_dataloader, svhn_dataloader, Lambda)

		print("Evaluating ...")
		valid_ac = model_evaluate(model, mnistm_test_dataloader)

		print("======================================================")

		if valid_ac > max_accuracy:
			max_accuracy = valid_ac
			torch.save(model.state_dict(), save_dir + "model_"+str(valid_ac))

	model = model.cpu()
