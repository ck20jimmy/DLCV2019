from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np
import pandas as pd


class ImageDataset(Dataset):

	def __init__(self, data_dir, label_path, transform):
        
		self.data_dir = data_dir
		self.label_path = label_path
		self.img_files = None
		self.all_labels = None
		self.transform = transform
		self.initFile()
		self.mean = np.array([0.44, 0.44, 0.44]).reshape(3, 1)
		self.std = np.array([0.19, 0.19, 0.19]).reshape(3, 1)

	def initFile(self):
		fn = os.listdir(self.data_dir)
		fn = [ f for f in fn if f.split('.')[-1] == 'png' ]
		self.img_files = sorted(fn, key=lambda x:int(x.split('.')[0].split('_')[0]) )
		# self.all_labels = pd.read_csv(self.label_path)
		# self.all_labels = {  self.all_labels.iloc[i,0]:self.all_labels.iloc[i,1] for i in range(self.all_labels.shape[0]) }
		

	def normalize(self, x):
		return (x-0.5)*2


	def standardize(self, x):
		return (x - self.mean) / self.std


	def __getitem__(self, index):

		fn = self.img_files[index]
		x = cv2.cvtColor(cv2.imread(self.data_dir + fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		x = x.transpose((2, 0, 1)).astype(np.float64) / 255.0

		x_norm = self.normalize(x)
		x_norm_ten = torch.FloatTensor(x_norm)

		stand_x = self.standardize(x.reshape(3, -1)).reshape(3, 28, 28)
		x_ten = torch.FloatTensor(stand_x)

		return x_ten, x_norm_ten#, label_ten


	def __len__(self):
		return len(self.img_files)
