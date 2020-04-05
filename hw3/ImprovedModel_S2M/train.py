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
import itertools



SOURCE_LABEL = 0
TARGET_LABEL = 1

def model_train(model, model_optim, source_dataloader, target_dataloader, curr_iter_init):

	class_criterion = nn.NLLLoss()
	real_fake_criterion = nn.BCELoss()

	fe_optim, source_optim, gen_optim, dis_optim = model_optim

	feat_extracter = model.feat_extracter.train()
	source_classifier = model.source_classifier.train()
	generater = model.generater.train()
	discriminator = model.discriminator.train()

	# trange = tqdm(zip(source_dataloader, target_dataloader))
	min_dataloader = min(len(source_dataloader), len(target_dataloader))

	source_iter = iter(source_dataloader)
	target_iter = iter(target_dataloader)

	trange = tqdm(range(min_dataloader))

	curr_iter = curr_iter_init

	# for idx, (source_data, target_data) in enumerate(trange):
	for idx in trange:

		# Train Feature Extractor
		
		# source data
		source_inp, source_norm, source_class = source_iter.next()
		source_inp = source_inp.cuda()
		source_norm = source_norm.cuda()
		source_class = source_class.cuda().view(-1)
		source_batch_size = source_inp.size(0)

		source_class_emb = torch.zeros( (source_batch_size, 10+1) ).float().cuda()
		source_class_emb[np.arange(source_batch_size), source_class] = 1

		#target data
		target_inp, target_norm, target_class = target_iter.next()
		target_inp = target_inp.cuda()
		target_norm = target_norm.cuda()
		target_class = target_class.cuda().view(-1)
		target_batch_size = target_inp.size(0)

		target_class_emb = torch.zeros( (target_batch_size, 10+1) ).float().cuda()
		target_class_emb[:, -1] = 1

		# optimize discriminator
		dis_optim.zero_grad()

		source_real_label = torch.ones(source_batch_size).cuda()
		source_fake_label = torch.zeros(source_batch_size).cuda()
		
		target_fake_label = torch.zeros(target_batch_size).cuda()
		target_real_label = torch.ones(target_batch_size).cuda()

		# source
		source_feat = feat_extracter(source_inp)

		source_gen_inp = torch.cat([source_class_emb, source_feat], dim=1)
		source_noise = torch.randn(source_batch_size, 512).cuda()
		source_gen_img = generater(source_gen_inp, source_noise)


		# target
		target_feat = feat_extracter(target_inp)

		target_gen_inp = torch.cat([target_class_emb, target_feat], dim=1)
		target_noise = torch.randn(target_batch_size, 512).cuda()
		target_gen_img = generater(target_gen_inp, target_noise)


		# source real
		source_real_class_prob, source_real_prob = discriminator(source_norm)
		errD_source_real = real_fake_criterion(source_real_prob, source_real_label)
		errD_source_real_class = class_criterion(source_real_class_prob, source_class)

		# source fake
		source_fake_class_prob, source_fake_prob = discriminator(source_gen_img)
		errD_source_fake = real_fake_criterion(source_fake_prob, source_fake_label)

		# target fake
		target_fake_class_prob, target_fake_prob = discriminator(target_gen_img)
		errD_target_fake = real_fake_criterion(target_fake_prob, target_fake_label)

		errD = errD_source_real + errD_source_real_class  + errD_source_fake + errD_target_fake
		errD.backward(retain_graph=True)
		dis_optim.step()
		

		# optimize generater
		gen_optim.zero_grad()

		source_fake_class_prob, source_fake_prob = discriminator(source_gen_img)
		errG_source_fake = real_fake_criterion(source_fake_prob, source_real_label)
		errG_source_fake_class = class_criterion(source_fake_class_prob, source_class)

		errG = errG_source_fake_class + errG_source_fake
		errG.backward(retain_graph=True)
		gen_optim.step()	
		

		# optimize source classifier
		source_optim.zero_grad()
		source_class_prob = source_classifier(source_feat)
		errSC = class_criterion(source_class_prob, source_class)
		errSC.backward(retain_graph=True)
		source_optim.step()

		source_ac = (F.softmax(source_class_prob, dim=1).argmax(dim=1).view(-1) == source_class.view(-1)).sum().float()
		source_ac /= source_batch_size

		with torch.no_grad():
			target_class_prob = source_classifier(target_feat.detach())
			target_ac = (F.softmax(target_class_prob, dim=1).argmax(dim=1).view(-1) == target_class.view(-1)).sum().float()
			target_ac /= target_batch_size


		# optimize feature extracter
		fe_optim.zero_grad()

		adv_weight = 0.1
		alpha = 0.3

		errF_SC = class_criterion(source_class_prob, source_class)

		source_fake_class_prob, source_fake_prob = discriminator(source_gen_img)
		errF_source_D_class = class_criterion(source_fake_class_prob, source_class) * adv_weight

		target_fake_class_prob, target_fake_prob = discriminator(target_gen_img)
		errF_target_D_fake = real_fake_criterion(target_fake_prob, target_real_label)
		errF_target_D_fake = errF_target_D_fake * adv_weight * alpha

		errF = errF_SC + errF_source_D_class + errF_target_D_fake
		errF.backward()
		fe_optim.step()

		trange.set_postfix(source_ac=source_ac.item(), target_ac=target_ac.item(), \
			errD=errD.item(), errG=errG.item(), \
			errSC=errSC.item(), errF=errF.item())

		curr_iter += 1

		# dis_optim = exp_lr_scheduler(dis_optim, 5e-4, 1e-4, curr_iter)    
		# fe_optim = exp_lr_scheduler(fe_optim, 5e-4, 1e-4, curr_iter)
		# source_optim = exp_lr_scheduler(source_optim, 5e-4, 1e-4, curr_iter)  

	return curr_iter


def exp_lr_scheduler(optimizer, init_lr, lrd, nevals):
	lr = init_lr / (1 + nevals*lrd)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer



def model_evaluate(feat_extracter, source_classifier, test_dataloader):

	feat_extracter = feat_extracter.eval()
	source_classifier = source_classifier.eval()

	trange = tqdm(test_dataloader)

	ac = 0.0
	all_data_num = 0.0

	with torch.no_grad():
		
		for data in trange:

			inp_img, inp_img_norm, inp_class = data

			inp_img = inp_img.cuda()
			inp_class = inp_class.cuda().view(-1)

			batch_size = inp_img.shape[0]

			test_feat = feat_extracter(inp_img)
			class_prob = source_classifier(test_feat)

			class_pred = class_prob.argmax(dim=1).view(-1)
			ac += (class_pred == inp_class).sum().item()

			all_data_num += batch_size

	ac /= all_data_num

	print("Evaluation Accuracy:{:.4}".format(ac))

	return ac


