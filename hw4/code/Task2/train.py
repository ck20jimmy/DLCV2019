import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from tqdm import tqdm


from data import MAX_LENGTH

def train(model, optim, train_data):

	model = model.train()

	trange = tqdm(train_data)

	criterion = nn.NLLLoss()

	total_loss = 0.0
	total_ac = 0.0
	total_data_num = 0.0

	optim.zero_grad()

	for idx, data in enumerate(trange):

		# if (idx+1) % 8 == 0:
		# 	optim.step()
		# 	optim.zero_grad()

		optim.zero_grad()

		video_ten, video_len_ten, label_ten = data

		batch_size = video_ten.size(0)

		video_ten = video_ten.cuda().view(-1, 3, 240, 320)
		video_len_ten = video_len_ten.cuda()
		label_ten = label_ten.cuda()

		video_feat = model.extract(video_ten).view(batch_size, MAX_LENGTH, 256)

		prob = model.encode(video_feat, video_len_ten)

		ac = (prob.argmax(dim=1).view(-1) == label_ten.view(-1)).sum().float()
		total_ac += ac.item()

		ac /= prob.size(0)

		loss = criterion(prob, label_ten)

		total_loss += loss.item()

		loss.backward()
		optim.step()

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

			batch_size = video_ten.size(0)

			video_ten = video_ten.cuda().view(-1, 3, 240, 320)
			video_len_ten = video_len_ten.cuda()
			label_ten = label_ten.cuda()

			video_feat = model.extract(video_ten).view(batch_size, MAX_LENGTH, 256)

			prob = model.encode(video_feat, video_len_ten)

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

			video_ten = video_ten.cuda().view(-1, 3, 240, 320)
			video_len_ten = video_len_ten.cuda()
			
			batch_size = video_len_ten.shape[0]

			video_feat = model.extract(video_ten).view(batch_size, MAX_LENGTH, 256)

			prob = model.encode(video_feat, video_len_ten)

			pred = prob.argmax(dim=1)

			all_pred.append(pred.cpu().numpy())

	all_pred = np.stack(all_pred)
	return all_pred.reshape(-1)