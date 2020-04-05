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


if __name__ == '__main__':

	train_data = ImageDataset("../../hw3_data/face/clean_data_dlib_cv2/", "../../hw3_data/face/train.csv")
	train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

	model = ACGAN()

	beta1 = 0.5
	beta2 = 0.9
	gen_optim = torch.optim.Adam(model.Generator.parameters(), lr=2e-4, betas=(beta1, beta2))
	dis_optim = torch.optim.Adam(model.Discriminator.parameters(), lr=2e-4, betas=(beta1, beta2))

	epochs = 2000

	save_dir = "../../model/ACGAN_DC/"

	fixed_inp = torch.zeros(64, 127)
	fixed_noise = torch.randn(32, 127)
	fixed_inp[:32] = fixed_noise
	fixed_inp[32:] = fixed_noise


	fixed_noise_smile = torch.zeros(64).view(64, 1)
	fixed_noise_smile[:32] = 1


	model = model.cuda()
	for ep in range(epochs):

		print("Training ...")
		model_train(model, gen_optim, dis_optim, train_dataloader)

		print("Evaluating ...")
		model_evaluate(model, save_dir + "eval_" + str(ep+1) + ".png", fixed_inp, fixed_noise_smile)

		print("======================================================")

		if (ep+1) % 20 == 0:
			torch.save(model.state_dict(), save_dir + "model_"+str(ep+1))

	model = model.cpu()
