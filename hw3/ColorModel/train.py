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


L1_LAMBDA = 100.0


def model_train(model, gen_optim, dis_optim, dataloader):

	
	gan_criterion = nn.BCELoss()
	sim_criterion = nn.L1Loss()

	trange = tqdm(dataloader)

	model = model.train()

	for idx, data in enumerate(trange):

		dis_optim.zero_grad()

		gray_img, color_img = data

		batch_size = gray_img.size(0)

		gray_img = gray_img.cuda()
		color_img = color_img.cuda()

		# Train Discriminator

		# All Real Label
		real_dis_inp = torch.cat([gray_img, color_img], dim=1)

		label = torch.ones(batch_size)
		label = label.cuda().view(-1)

		real_prob = model.discriminate(real_dis_inp).view(-1)
		err_dis_real = gan_criterion(real_prob, label)
		err_dis_real.backward()

		D_x = real_prob.mean().item()


		# All Fake Label
		fake_color_img = model.generate(gray_img)
		label = torch.zeros(batch_size).view(-1).cuda()

		fake_dis_inp = torch.cat([gray_img, fake_color_img.detach()], dim=1)

		fake_prob = model.discriminate(fake_dis_inp).view(-1)
		err_dis_fake = gan_criterion(fake_prob, label)
		err_dis_fake.backward()

		D_G_z1 = fake_prob.mean().item()

		err_dis = err_dis_fake + err_dis_real
		dis_optim.step()


		# Train Generator
		gen_optim.zero_grad()

		label = torch.ones(batch_size).view(-1).cuda()

		gen_inp = torch.cat([gray_img, fake_color_img], dim=1)
		gen_prob = model.discriminate(gen_inp).view(-1)

		err_gen_gan = gan_criterion(gen_prob, label)
		err_gen_sim = sim_criterion(fake_color_img, color_img) * L1_LAMBDA

		err_gen = err_gen_gan + err_gen_sim
		err_gen.backward()

		D_G_z2 = gen_prob.mean().item()

		gen_optim.step()

		trange.set_postfix(errD=err_dis.item(), errG=err_gen.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)


def model_evaluate(model, inp_img, fn):

	model = model.eval()

	with torch.no_grad():
		
		fake_image = model.generate(inp_img).detach()
		
		fake_image = (fake_image + 1) / 2 * 255.0

		l_image = (inp_img + 1) / 2 * 255.0

		lab_image = torch.cat((l_image, fake_image), dim=1).cpu().numpy().transpose((0,2,3,1))

		for i in range(lab_image.shape[0]):
			lab_image[i] = cv2.cvtColor(lab_image[i].astype(np.uint8), cv2.COLOR_LAB2RGB)

		# cv2.imwrite(fn, lab_image[0])

		out_image = torch.from_numpy( lab_image.transpose(0, 3, 1, 2) )

		save_image(out_image, fn, nrow=8, normalize=True)