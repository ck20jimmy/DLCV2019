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

		optim.zero_grad()

		video_ten, video_mask_ten, label_ten = data

		batch_size = video_ten.size(0)

		video_ten = video_ten.cuda().view(-1, 3, 240, 320)
		video_mask_ten = video_mask_ten.cuda()
		label_ten = label_ten.cuda()

		video_feat = model.extract(video_ten).view(batch_size, MAX_LENGTH, 256)
		
		loss = 0.0

		batch_ac = 0.0
		for i in range(MAX_LENGTH):
			emb = model.encode(video_feat[:,i,:].unsqueeze(1)).squeeze()
			prob = model.classify(emb)

			ac = (prob.argmax(dim=1).view(-1) == label_ten[:,i].view(-1)) * video_mask_ten[:,i].view(-1)
			batch_ac += ac.sum().float()

			l = criterion( prob[video_mask_ten[:,i].view(-1),:], \
				label_ten[video_mask_ten[:,i].view(-1),i] )
			loss += l

		total_ac += batch_ac.sum().item()
		batch_ac /= video_mask_ten.sum()

		total_loss += loss.item()

		loss.backward()
		optim.step()

		avg_loss = loss / MAX_LENGTH

		trange.set_postfix(Loss=avg_loss.item(), Accuracy=batch_ac.item())

		total_data_num += video_mask_ten.sum().item()

	total_loss /= total_data_num
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

		for idx, data in enumerate(trange):

			video_ten, video_mask_ten, label_ten = data

			batch_size = video_ten.size(0)

			video_ten = video_ten.cuda().view(-1, 3, 240, 320)
			video_mask_ten = video_mask_ten.cuda()
			label_ten = label_ten.cuda()

			video_feat = model.extract(video_ten).view(batch_size, MAX_LENGTH, 256)
			
			loss = 0.0

			batch_ac = 0.0
			for i in range(MAX_LENGTH):
				emb = model.encode(video_feat[:,i,:].unsqueeze(1)).squeeze()
				prob = model.classify(emb)
				ac = (prob.argmax(dim=1).view(-1) == label_ten[:,i].view(-1)) * video_mask_ten[:,i].view(-1)
				batch_ac += ac.sum().float()

				l = criterion( prob[video_mask_ten[:,i].view(-1),:], \
					label_ten[video_mask_ten[:,i].view(-1),i] )
				loss += l

			total_ac += batch_ac.sum().item()
			batch_ac /= video_mask_ten.sum()

			total_loss += loss.item()

			avg_loss = loss / MAX_LENGTH

			trange.set_postfix(Loss=avg_loss.item(), Accuracy=batch_ac.item())

			total_data_num += video_mask_ten.sum().item()

	total_loss /= total_data_num
	total_ac /= total_data_num

	return total_loss, total_ac




def predict(model, valid_data):

	model = model.eval()

	trange = tqdm(valid_data)

	all_pred = []

	with torch.no_grad():

		for idx, data in enumerate(trange):

			video_ten, video_mask_ten = data

			batch_size = video_ten.size(0)

			video_ten = video_ten.cuda().view(-1, 3, 240, 320)
			video_mask_ten = video_mask_ten.cuda()

			video_feat = model.extract(video_ten).view(batch_size, MAX_LENGTH, 256)

			batch_pred = []

			for i in range(MAX_LENGTH):
				emb = model.encode(video_feat[:,i,:].unsqueeze(1)).squeeze()
				prob = model.classify(emb)
				pred = prob.argmax(dim=1).cpu().numpy()#.masked_select(video_mask_ten[:,i]).cpu().numpy()
				batch_pred.append(pred)

			batch_pred = np.stack(batch_pred).transpose((1,0))

			batch_pred = batch_pred[video_mask_ten.cpu().numpy()]

			all_pred.append(batch_pred)

	all_pred = np.concatenate(all_pred, axis=0)

	return all_pred