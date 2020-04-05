import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import sys

from model import *
from train import *
from reader import *

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models


class TrimImageDataset(Dataset):

	def __init__(self, video_path, label_path):

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

		video_cat = self.image_dict["Video_category"][index]
		video_name = self.image_dict["Video_name"][index]
		
		video = readShortVideo(self.video_path, video_cat, video_name, downsample_factor=12, rescale_factor=1)
		norm_video = self.normalize(video.astype(np.float64) / 255.0).transpose((0, 3, 1, 2))

		#sample the middle 
		if norm_video.shape[0] > 10:
			mid = norm_video.shape[0] // 2
			norm_video = norm_video[mid-5:mid+5]

		video_len = norm_video.shape[0]
		pad_len = 10 - video_len
		pad_emb = np.zeros( (pad_len, 3, 240, 320) )

		norm_video = np.concatenate((norm_video, pad_emb), axis=0)

		return norm_video, video_len

	def __len__(self):
		return len(self.image_dict["Video_index"])



def collate_fn(data):

	video, video_len = zip(*data)
	
	video = np.stack(video)
	video_len = np.stack(video_len)

	sort_idx = video_len.argsort(axis=0)[::-1]

	video = video[sort_idx]
	video_len = video_len[sort_idx]

	video_len_ten = torch.from_numpy(video_len).long()
	video_ten = torch.from_numpy(video).float()

	return video_ten, video_len_ten



if __name__ == '__main__':

    test_video_dir = sys.argv[1] + '/'
    test_label_path = sys.argv[2]

    out_dir = sys.argv[3] + '/'
    model_path = sys.argv[4]

    # valid_dataset = TrimImageDataset("../../hw4_data/TrimmedVideos/video/valid/")
    valid_dataset = TrimImageDataset(test_video_dir, test_label_path)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1, num_workers=3, collate_fn=collate_fn)

    pretrain = models.resnet50(pretrained=True)
    model = Model(pretrain)

    model.load_state_dict(torch.load(model_path))

    model = model.cuda()

    pred = predict(model, valid_dataloader)

    with open(out_dir + 'p2_result.txt', 'w') as fd:
        for i in range(pred.shape[0]):
            if i != 0:
                fd.write('\n')
            fd.write(str(pred[i]))
        fd.close()
