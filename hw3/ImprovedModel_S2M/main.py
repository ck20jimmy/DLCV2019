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


# http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Generate_to_Adapt_CVPR_2018_paper.pdf
# https://github.com/yogeshbalaji/Generate_To_Adapt


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		size = m.weight.size()
		m.weight.data.normal_(0.0, 0.1)
		m.bias.data.fill_(0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		nn.init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		nn.init.normal(m.weight.data, 1.0, 0.02)
		nn.init.constant(m.bias.data, 0.0)



if __name__ == '__main__':

	# transform = trns.Compose([ trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
	transform = trns.Compose([])

	mnist_data = ImageDataset("../../hw3_data/digits/mnistm/train/", \
		"../../hw3_data/digits/mnistm/train.csv", \
		transform )
	mnist_dataloader = DataLoader(mnist_data, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

	svhn_data = ImageDataset("../../hw3_data/digits/svhn/train/", \
		"../../hw3_data/digits/svhn/train.csv", \
		transform )
	svhn_dataloader = DataLoader(svhn_data, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

	mnistm_testdata = ImageDataset("../../hw3_data/digits/mnistm/test/", \
		"../../hw3_data/digits/mnistm/test.csv", \
		transform )
	mnistm_test_dataloader = DataLoader(mnistm_testdata, batch_size=128, shuffle=False, num_workers=2, drop_last=False)


	model = GTA()

	model.feat_extracter.apply(weights_init)
	model.source_classifier.apply(weights_init)
	model.generater.apply(weights_init)
	model.discriminator.apply(weights_init)


	learning_rate = 5e-4
	fe_optim = torch.optim.Adam(model.feat_extracter.parameters(), lr=learning_rate, betas=(0.8, 0.999))
	source_optim = torch.optim.Adam(model.source_classifier.parameters(), lr=learning_rate, betas=(0.8, 0.999))
	gen_optim = torch.optim.Adam(model.generater.parameters(), lr=learning_rate, betas=(0.8, 0.999))
	dis_optim = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0.8, 0.999))

	model_optim = (fe_optim, source_optim, gen_optim, dis_optim)

	epochs = 100

	save_dir = "../../model/ImprovedModel/Svhn2MnistM/Svhn2MnistM/"

	max_accuracy = -1

	model = model.cuda()

	curr_iter = 0

	for ep in range(epochs):

		print("Epoch {}".format(ep+1))

		print("Training ...")
		curr_iter = model_train(model, model_optim, svhn_dataloader, mnist_dataloader, curr_iter)
		# curr_iter = model_train(model, model_optim, mnist_dataloader, svhn_dataloader, curr_iter)

		print("Evaluating ...")
		valid_ac = model_evaluate(model.feat_extracter, model.source_classifier, mnistm_test_dataloader)

		print("======================================================")

		if valid_ac > max_accuracy:
			max_accuracy = valid_ac
			torch.save(model.state_dict(), save_dir + "model_"+str(valid_ac))

	model = model.cpu()
