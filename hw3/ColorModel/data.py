from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np

class ImageDataset(Dataset):


	def __init__(self, data_dir):
        
		self.data_dir = data_dir
		self.img_files = None
		self.initFile()

	def initFile(self):
		fn = os.listdir(self.data_dir)
		fn = [ f for f in fn if f.split('.')[-1] == 'png' ]
		self.img_files = sorted(fn, key=lambda x:int(x.split('.')[0].split('_')[0]) )


	# def normalize(self, x):
	# 	channel_num = x.shape[0]
	# 	x_min = x.reshape(channel_num, -1).min(axis=1).reshape(channel_num,1)
	# 	x_max = x.reshape(channel_num, -1).max(axis=1).reshape(channel_num,1)
		
	# 	ran21 = (x_max - x_min) / 2.0

	# 	out = (x.reshape(channel_num, -1) - ran21 - x_min) / ran21
	# 	out = out.reshape(x.shape)

	# 	return out

	def normalize(self, x):
		out = (x / 255.0) * 2 - 1
		return out


	def __getitem__(self, index):

		fn = self.img_files[index]
		img = cv2.imread(self.data_dir + fn, cv2.IMREAD_COLOR)

		lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		# x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
		x = lab_img[:,:,0].astype(np.float64)
		x = x.reshape(1, 64, 64)
		norm_x = self.normalize(x)

		x_ten = torch.FloatTensor(norm_x)

		# y = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
		y = lab_img[:,:,1:].astype(np.float64)
		y = y.transpose((2, 0, 1))
		norm_y = self.normalize(y)

		y_ten = torch.FloatTensor(norm_y)

		return x_ten, y_ten

	def __len__(self):
		return len(self.img_files)



