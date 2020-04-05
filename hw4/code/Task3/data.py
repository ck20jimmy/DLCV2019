from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np
import pandas as pd
import math

MAX_LENGTH = 20

class FullLengthDataset(Dataset):

	def __init__(self, label_dir, video_dir):

		self.video_dir = video_dir
		self.label_dir = label_dir

		self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 3)
		self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 3)

		self.prepareImage()

	def prepareImage(self):

		all_label_file = [ f for f in os.listdir(self.label_dir) if f.split('.')[1] == 'txt']
		
		all_image_file = []
		all_label = []

		for fn in all_label_file:
			video_name = fn.split('.')[0]

			image_fn = [ f for f in os.listdir(self.video_dir + '/' + video_name) if f.split('.')[1] == 'jpg']
			image_fn = [ os.path.join(self.video_dir, video_name, f) \
				for f in sorted(image_fn, key=lambda x:int(x.split('.')[0])) ]

			label = []
			with open(self.label_dir + '/' + fn, 'r') as fd:
				for line in fd:
					tmp = line.strip('\n ')
					label.append(int(tmp))
			
			for i in range(len(label) // MAX_LENGTH):
				all_label.append(label[ i*MAX_LENGTH:(i+1)*MAX_LENGTH ])
				all_image_file.append(image_fn[ i*MAX_LENGTH:(i+1)*MAX_LENGTH ])

		self.labels = all_label
		self.image_file = all_image_file


	def normalize(self, video):
		H, W, C = video.shape
		norm_video = (video.reshape(-1, C) - self.mean) / self.std
		return norm_video.reshape((H, W, C))

	def readVideo(self, img_fn_list):
		images = []
		
		for fn in img_fn_list:
			img_bgr = cv2.imread(fn)
			img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
			norm_img = self.normalize(img)
			images.append(norm_img)
		
		return np.stack(images)

	def __getitem__(self, index):
		
		video = self.readVideo(self.image_file[index]).transpose((0, 3, 1, 2))
		video_len = video.shape[0]

		label = np.array(self.labels[index]).astype(int)
		mask = np.zeros(MAX_LENGTH).astype(int)
		mask[:video.shape[0]] = 1

		pad_emb = np.zeros( (MAX_LENGTH - video_len, 3, video.shape[2], video.shape[3]) )
		video = np.concatenate((video, pad_emb), axis=0)

		pad_emb = np.zeros(MAX_LENGTH - video_len)
		label = np.concatenate( (label, pad_emb), axis=0)

		return video, mask , label

	def __len__(self):
		return len(self.image_file)
