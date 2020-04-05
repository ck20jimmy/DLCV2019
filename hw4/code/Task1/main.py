import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import sys

from data import *
from model import *
from train import *


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models


def collate_fn(data):

	video, video_len, video_label = zip(*data)

	video_ten = torch.from_numpy(np.concatenate(video, axis=0)).float()
	video_len_ten = torch.from_numpy(np.stack(video_len)).long()
	video_label_ten = torch.from_numpy(np.stack(video_label)).long()

	return video_ten, video_len_ten, video_label_ten



if __name__ == '__main__':

	train_dataset = TrimImageDataset("../../hw4_data/TrimmedVideos/label/gt_train.csv", \
		"../../hw4_data/TrimmedVideos/video/train/")
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=3, collate_fn=collate_fn)

	valid_dataset = TrimImageDataset("../../hw4_data/TrimmedVideos/label/gt_valid.csv", \
		"../../hw4_data/TrimmedVideos/video/valid/")
	valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=16, num_workers=3, collate_fn=collate_fn)

	pretrain = models.resnet50(pretrained=True)
	model = Model(pretrain)
	
	model = model.cuda()

	epochs = 100

	optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

	min_valid_loss = 2147483647.0

	train_losses = []
	valid_losses = []

	train_acs = []
	valid_acs = []

	for ep in range(epochs):

		print("Epoch {}".format(ep+1))
		
		train_loss, train_ac = train(model, optimizer, train_dataloader)
		print("Training Loss:{}\tTraining Accuracy:{}".format(train_loss, train_ac))
		
		train_losses.append(train_loss)
		train_acs.append(train_ac)

		valid_loss, valid_ac = evaluate(model, valid_dataloader)
		print("Validaiton Loss:{}\tValidation Accuracy:{}".format(valid_loss, valid_ac))

		if valid_loss < min_valid_loss:
			min_valid_loss = valid_loss
			torch.save(model.state_dict(), "../../model/Task1/best_model")

		valid_losses.append(valid_loss)
		valid_acs.append(valid_ac)

		scheduler.step(valid_loss)