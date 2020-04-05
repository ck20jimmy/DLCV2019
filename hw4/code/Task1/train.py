import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from tqdm import tqdm




def train(model, optim, train_data):

	model = model.train()

	trange = tqdm(train_data)

	criterion = nn.NLLLoss()

	total_loss = 0.0
	total_ac = 0.0
	total_data_num = 0.0

	optim.zero_grad()

	for idx, data in enumerate(trange):

		if (idx+1) % 8 == 0:
			optim.step()
			optim.zero_grad()

		# optim.zero_grad()

		video_ten, video_len_ten, label_ten = data

		video_ten = video_ten.cuda()
		video_len_ten = video_len_ten.cuda()
		label_ten = label_ten.cuda()

		video_feat = model.extract(video_ten)

		# average pooling
		current = 0
		pooling_feat = []
		for l in video_len_ten:
			pooling_feat.append(video_feat[current:current+l].mean(dim=0))
			current += l

		video_feat = video_feat.cpu()
		pooling_feat = torch.stack(pooling_feat).cuda()

		prob = model.classify(pooling_feat)

		ac = (prob.argmax(dim=1).view(-1) == label_ten.view(-1)).sum().float()
		total_ac += ac.item()

		ac /= prob.size(0)

		loss = criterion(prob, label_ten)

		total_loss += loss.item()

		loss.backward()
		# optim.step()

		trange.set_postfix(Loss=loss.item(), Accuracy=ac.item())

		total_data_num += prob.size(0)

	total_loss /= len(train_data)
	total_ac /= total_data_num

	return total_loss, total_ac



def evaluate(model, valid_data):

	model = model.eval()

	trange = tqdm(valid_data)

	criterion = nn.NLLLoss()

	total_loss = 0.0
	total_ac = 0.0
	total_data_num = 0.0

	with torch.no_grad():

		for data in trange:

			video_ten, video_len_ten, label_ten = data

			video_ten = video_ten.cuda()
			video_len_ten = video_len_ten.cuda()
			label_ten = label_ten.cuda()

			video_feat = model.extract(video_ten)

			# average pooling
			current = 0
			pooling_feat = []
			for l in video_len_ten:
				pooling_feat.append(video_feat[current:current+l].mean(dim=0))
				current += l

			pooling_feat = torch.stack(pooling_feat).cuda()

			prob = model.classify(pooling_feat)

			ac = (prob.argmax(dim=1).view(-1) == label_ten.view(-1)).sum().float()
			total_ac += ac.item()

			ac /= prob.size(0)

			loss = criterion(prob, label_ten)

			total_loss += loss.item()

			trange.set_postfix(Loss=loss.item(), Accuracy=ac.item())

			total_data_num += prob.size(0)

	total_loss /= len(valid_data)
	total_ac /= total_data_num

	return total_loss, total_ac


def predict(model, valid_data):

	model = model.eval()

	trange = tqdm(valid_data)

	all_pred = []

	with torch.no_grad():

		for data in trange:

			video_ten, video_len_ten = data

			video_ten = video_ten.cuda()
			video_len_ten = video_len_ten.cuda()

			video_feat = model.extract(video_ten)

			# average pooling
			current = 0
			pooling_feat = []
			for l in video_len_ten:
				pooling_feat.append(video_feat[current:current+l].mean(dim=0))
				current += l

			pooling_feat = torch.stack(pooling_feat).cuda()

			prob = model.classify(pooling_feat)
			pred = prob.argmax(dim=1)

			all_pred.append(pred.cpu().numpy().tolist())


	return np.stack(all_pred)