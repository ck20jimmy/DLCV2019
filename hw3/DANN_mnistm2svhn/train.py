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

SOURCE_LABEL = 0
TARGET_LABEL = 1

def model_train(model, model_optim, source_dataloader, target_dataloader, Lambda):

	class_criterion = nn.NLLLoss()
	domain_criterion = nn.BCELoss()

	min_dataloader = min(len(source_dataloader), len(target_dataloader))

	source_iter = iter(source_dataloader)
	target_iter = iter(target_dataloader)

	model = model.train()

	trange = tqdm(range(min_dataloader))

	for idx in trange:

		# train the domain classifier

		# source data

		# Train Feature Extractor
		model_optim.zero_grad()
		
		# source data
		source_inp, source_class = source_iter.next()
		source_inp = source_inp.cuda()
		source_class = source_class.cuda().view(-1)

		source_batch_size = source_inp.size(0)

		class_prob, domain_prob = model.fit(source_inp, Lambda)

		# train class classifier
		source_class_loss = class_criterion(class_prob, source_class)

		class_pred = class_prob.argmax(dim=1).view(-1)
		source_ac = (class_pred == source_class).sum().float()
		source_ac /= source_batch_size

		domain_label = torch.ones(source_batch_size) * SOURCE_LABEL
		domain_label = domain_label.cuda().view(-1, 1)
		source_domain_loss = domain_criterion(domain_prob, domain_label)


		#target data
		target_inp, target_class = target_iter.next()
		target_inp = target_inp.cuda()
		target_class = target_class.cuda().view(-1)

		target_batch_size = target_inp.size(0)

		class_prob, domain_prob = model.fit(target_inp, Lambda)

		# train class classifier

		target_class_loss = class_criterion(class_prob, target_class)

		class_pred = class_prob.argmax(dim=1).view(-1)
		target_ac = (class_pred == target_class).sum().float()
		target_ac /= target_batch_size

		domain_label = torch.ones(target_batch_size) * TARGET_LABEL
		domain_label = domain_label.cuda().view(-1, 1)
		target_domain_loss = domain_criterion(domain_prob, domain_label)

		class_loss = source_class_loss
		domain_loss = source_domain_loss + target_domain_loss

		total_loss = class_loss + domain_loss

		total_loss.backward()

		model_optim.step()


		trange.set_postfix(source_ac=source_ac.item(), target_ac=target_ac.item(), \
			class_loss=class_loss.item(), domain_loss=domain_loss.item(), \
		)

		# trange.set_postfix(source_ac=source_ac.item(), target_ac=target_ac.item(), class_loss=class_loss.item() )

		


def model_evaluate(model, test_dataloader):

	model = model.eval()

	trange = tqdm(test_dataloader)

	ac = 0.0
	all_data_num = 0.0

	with torch.no_grad():
		
		for data in trange:

			inp_img, inp_class = data

			inp_img = inp_img.cuda()
			inp_class = inp_class.cuda().view(-1, 1)

			batch_size = inp_img.shape[0]

			class_prob = model.predict(inp_img)

			class_pred = class_prob.argmax(dim=1).view(-1, 1)
			ac += (class_pred == inp_class).sum().item()

			all_data_num += batch_size

	ac /= all_data_num

	print("Evaluation Accuracy:{:.4}".format(ac))

	return ac