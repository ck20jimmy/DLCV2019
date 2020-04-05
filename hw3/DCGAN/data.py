from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np

from train import INPUT_CHANNEL

class ImageDataset(Dataset):


	def __init__(self, data_dir):
        
		self.data_dir = data_dir
		self.img_files = None
		self.initFile()

	def initFile(self):
		fn = os.listdir(self.data_dir)
		fn = [ f for f in fn if f.split('.')[-1] == 'png' ]
		self.img_files = sorted(fn, key=lambda x:int(x.split('.')[0].split('_')[0]) )

	def normalize(self, x):
		x_min = x.reshape(INPUT_CHANNEL, -1).min(axis=1).reshape(INPUT_CHANNEL,1)
		x_max = x.reshape(INPUT_CHANNEL, -1).max(axis=1).reshape(INPUT_CHANNEL,1)
		
		ran21 = (x_max - x_min) / 2.0

		out = (x.reshape(INPUT_CHANNEL, -1) - ran21 - x_min) / ran21
		out = out.reshape(x.shape)

		return out

	def __getitem__(self, index):

		fn = self.img_files[index]
		x = cv2.cvtColor(cv2.imread(self.data_dir + fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float64)
		x = x.reshape(INPUT_CHANNEL, 64, 64)

		norm_x = self.normalize(x)
		x_ten = torch.FloatTensor(norm_x)

		y_ten = torch.ones(1)

		return x_ten, y_ten

	def __len__(self):
		return len(self.img_files)
