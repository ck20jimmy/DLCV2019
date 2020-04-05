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

from torch.optim.lr_scheduler import ReduceLROnPlateau


# https://arxiv.org/pdf/1611.07004v1.pdf
# https://github.com/ImagingLab/Colorizing-with-GANs
# https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/
# https://github.com/pdrabinski/GAN_Colorizer

if __name__ == '__main__':


	train_data = ImageDataset("../../hw3_data/face/clean_data_dlib_cv2/")
	train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)


	model = ColorModel()

	beta1 = 0.5
	gen_optim = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(beta1, 0.999))
	dis_optim = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(beta1, 0.999))


	epochs = 100

	save_dir = "../../model/ColorModel2/"

	eval_img = []
	for i in range(64):
		black_img, color_img = train_data.__getitem__(i)
		eval_img.append(black_img)

	eval_img = torch.stack(eval_img)

	# model = (modelR.cuda(), modelG.cuda(), modelB.cuda())
	# model_optim = (modelR_optim, modelG_optim, modelB_optim)

	model = model.cuda()

	for ep in range(epochs):

		print("Training ...")
		model_train(model, gen_optim, dis_optim, train_dataloader)

		if (ep+1) % 1 == 0:
			print("Evaluating ...")
			eval_img = eval_img.cuda()
			model_evaluate(model, eval_img, save_dir + "eval_" + str(ep+1) + ".png")
			eval_img = eval_img.cpu()

		print("======================================================")

		if (ep+1) % 10 == 0:
			torch.save(model.state_dict(), save_dir + "model_"+str(ep+1))

	model = model.cpu()