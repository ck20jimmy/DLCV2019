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

		image_inp, smile_label = data

		batch_size = image_inp.size(0)

		image_inp = image_inp.cuda()
		smile_label = smile_label.cuda()
		real_fake_label = torch.ones(batch_size).cuda()


		# Train Discriminator

		# All Real Label
		real_prob, smile_prob = model.discriminate(image_inp)
		err_dis_real = criterion(real_prob, real_fake_label)
		err_dis_real_smile = criterion(smile_prob, smile_label)

		err_dis_real = err_dis_real + err_dis_real_smile
		err_dis_real.backward()

		D_x_real = real_prob.mean().item()
		D_x_real_smile = smile_prob.mean().item()


		# All Fake Label
		noise = torch.randn(batch_size, 127).cuda()
		noise_smile = torch.FloatTensor(batch_size).random_(0, 2).view(-1, 1).cuda()
		noise_label = torch.zeros(batch_size).cuda()

		fake_image = model.generate(noise, noise_smile)

		fake_prob, smile_prob = model.discriminate(fake_image.detach())
		
		err_dis_fake = criterion(fake_prob, noise_label)
		err_dis_fake_smile = criterion(smile_prob, noise_smile)

		err_dis_fake = err_dis_fake + err_dis_fake_smile
		err_dis_fake.backward()

		D_x_fake = fake_prob.mean().item()
		D_x_fake_smile = err_dis_fake_smile.mean().item()

		err_dis = err_dis_fake + err_dis_real

		D_x_label = (D_x_fake + D_x_real) / 2
		D_x_smile = (D_x_fake_smile + D_x_real_smile) / 2

		dis_optim.step()


		# Train Generator
		gen_optim.zero_grad()

		noise_label = torch.ones(batch_size).cuda()

		fake_prob, smile_prob = model.discriminate(fake_image)
		err_gen_fake = criterion(fake_prob, noise_label)
		err_gen_smile = criterion(smile_prob, noise_smile)

		err_gen = err_gen_fake + err_gen_smile

		err_gen.backward()
		
		D_G_label = fake_prob.mean().item()
		D_G_smile = smile_prob.mean().item()

		gen_optim.step()

		trange.set_postfix(errD=err_dis.item(), errG=err_gen.item(), \
			D_x_label=D_x_label, D_x_smile=D_x_smile, \
			D_G_label=D_G_label, D_G_smile=D_G_smile)




def model_evaluate(model, fn, noise, noise_class):

	model = model.eval()

	with torch.no_grad():
		
		noise = noise.cuda()
		noise_class = noise_class.cuda()

		fake_image = model.generate(noise, noise_class).detach().cpu()

		save_image(fake_image, fn, nrow=8, normalize=True)

