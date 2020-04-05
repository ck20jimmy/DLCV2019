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
from torchvision.utils import save_image



IMAGE_SIZE = 64
INPUT_CHANNEL = 1
NOISE_SIZE = 128



def model_train(model, gen_optim, dis_optim, dataloader):

	
	criterion = nn.BCELoss()

	trange = tqdm(dataloader)


	for idx, data in enumerate(trange):

		model = model.train()

		dis_optim.zero_grad()

		image_inp, label = data

		image_inp = image_inp.cuda()
		label = label.cuda().view(-1)

		batch_size = image_inp.size(0)

		# Train Discriminator

		# All Real Label
		output = model.discriminate(image_inp).view(-1)
		err_dis_real = criterion(output, label)
		err_dis_real.backward()

		D_x = output.mean().item()

		# All Fake Label

		# noise = torch.randn(batch_size, NOISE_SIZE, 1, 1)

		noise = torch.normal(mean=torch.zeros(NOISE_SIZE*batch_size), std=(torch.ones(NOISE_SIZE*batch_size)*1.0))
		noise = noise.reshape(batch_size, NOISE_SIZE, 1, 1)
		noise = noise.cuda()

		fake_image = model.generate(noise)
		label = torch.zeros(batch_size).view(batch_size).cuda()

		output = model.discriminate(fake_image.detach()).view(-1)
		err_dis_fake = criterion(output, label)
		err_dis_fake.backward()

		D_G_z1 = output.mean().item()

		err_dis = err_dis_fake + err_dis_real

		dis_optim.step()


		# Train Generator
		gen_optim.zero_grad()

		label = torch.ones(batch_size).view(-1).cuda()
		output = model.discriminate(fake_image).view(-1)
		err_gen = criterion(output, label)

		err_gen.backward()
		D_G_z2 = output.mean().item()

		gen_optim.step()

		trange.set_postfix(errD=err_dis.item(), errG=err_gen.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)

		# if (idx+1) % 200 == 0:
		# 	model = model.eval()

		# 	with torch.no_grad():

		# 		noise = torch.randn(16, NOISE_SIZE, 1, 1)
		# 		noise = noise.cuda()
		# 		fake_image = model.generate(noise).detach().cpu()

		# 		save_image(fake_image, "../model/GAN/test.png", nrow=4, normalize=True)



def model_evaluate(model, fn, noise):

	model = model.eval()

	with torch.no_grad():
		
		noise = noise.cuda()
		fake_image = model.generate(noise).detach().cpu()

		save_image(fake_image, fn, nrow=8, normalize=True)

