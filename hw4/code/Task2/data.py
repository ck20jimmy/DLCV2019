from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np
import pandas as pd

from reader import *

MAX_LENGTH = 10

class TrimImageDataset(Dataset):

	def __init__(self, label_path, video_path):

		self.video_path = video_path
		self.label_path = label_path
		self.image_dict = getVideoList(label_path)

		self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 3)
		self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 3)

	def normalize(self, video):
		T, H, W, C = video.shape
		norm_video = (video.reshape(-1, C) - self.mean) / self.std
		return norm_video.reshape((T, H, W, C))


	def __getitem__(self, index):

		label = int(self.image_dict["Action_labels"][index])
		video_cat = self.image_dict["Video_category"][index]
		video_name = self.image_dict["Video_name"][index]
		
		video = readShortVideo(self.video_path, video_cat, video_name, downsample_factor=12, rescale_factor=1)
		norm_video = self.normalize(video.astype(np.float64) / 255.0).transpose((0, 3, 1, 2))

		#sample the middle 
		if norm_video.shape[0] > MAX_LENGTH:
			mid = norm_video.shape[0] // 2
			norm_video = norm_video[mid-5:mid+5]

		video_len = norm_video.shape[0]
		pad_len = MAX_LENGTH - video_len
		pad_emb = np.zeros( (pad_len, 3, 240, 320) )

		norm_video = np.concatenate((norm_video, pad_emb), axis=0)

		return norm_video, video_len, label

	def __len__(self):
		return len(self.image_dict["Video_index"])
