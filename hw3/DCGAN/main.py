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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



if __name__ == '__main__':


	train_data = ImageDataset("../../hw3_data/face/clean_data_dlib_cv2/")
	train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=6)


	model = ContourGAN(NOISE_SIZE, IMAGE_SIZE, INPUT_CHANNEL)

	model.Generator.apply(weights_init)
	model.Discriminator.apply(weights_init)

	# model.load_state_dict(torch.load("../model/GAN/model_260"))
	# model.Generator.load_state_dict(torch.load("../model/GAN/model_260"))
	# model.Generator.load_state_dict(torch.load("../model/GAN/model_260"))

	beta1 = 0.5
	beta2 = 0.9
	gen_optim = torch.optim.Adam(model.Generator.parameters(), lr=2e-4, betas=(beta1, beta2))
	dis_optim = torch.optim.Adam(model.Discriminator.parameters(), lr=2e-4, betas=(beta1, beta2))

	epochs = 2000

	save_dir = "../../model/DCGAN/"

	#fixed_noise = torch.randn(64, NOISE_SIZE, 1, 1)
	fixed_noise = torch.normal(mean=torch.zeros(NOISE_SIZE*64), std=(torch.ones(NOISE_SIZE*64)*1.0))
	fixed_noise = fixed_noise.reshape(64, NOISE_SIZE, 1, 1)


	model = model.cuda()
	for ep in range(epochs):

		print("Training ...")
		model_train(model, gen_optim, dis_optim, train_dataloader)

		print("Evaluating ...")
		model_evaluate(model, save_dir + "eval_" + str(ep+1) + ".png", fixed_noise)

		print("======================================================")

		if (ep+1) % 40 == 0:
			torch.save(model.state_dict(), save_dir + "model_"+str(ep+1))

	model = model.cpu()
