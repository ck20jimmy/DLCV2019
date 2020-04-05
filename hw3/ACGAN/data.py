from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np
import pandas as pd

from train import INPUT_CHANNEL

class ImageDataset(Dataset):


	def __init__(self, data_dir, label_path):
        
		self.data_dir = data_dir
		self.label_path = label_path
		self.img_files = None
		self.all_labels = None
		self.initFile()

	def initFile(self):
		fn = os.listdir(self.data_dir)
		fn = [ f for f in fn if f.split('.')[-1] == 'png' ]
		self.img_files = sorted(fn, key=lambda x:int(x.split('.')[0].split('_')[0]) )
		self.all_labels = pd.read_csv(self.label_path)[["image_name", "Smiling"]]

	def normalize(self, x):
		out = (x / 255.0) * 2 - 1
		return out

	def __getitem__(self, index):

		fn = self.img_files[index]
		x = cv2.cvtColor(cv2.imread(self.data_dir + fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
		x = x[:,:,0]
		x = x.reshape(INPUT_CHANNEL, 64, 64).astype(np.float64)

		norm_x = self.normalize(x)
		x_ten = torch.FloatTensor(norm_x)

		label = self.all_labels[ self.all_labels['image_name'] == fn ]["Smiling"].item()
		label_ten = torch.FloatTensor([label])

		return x_ten, label_ten


	def __len__(self):
		return len(self.img_files)
