
import numpy as np
import sys
import os
from PIL import Image
import cv2
import pickle
from argparse import ArgumentParser


import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

from model import UNet, ResUNet

class ImageDataset(Dataset):


	def __init__(self, data_dir, transform):
        
		self.data_dir = data_dir
		self.img_files = None
		self.initFile()

		self.transform = transform

	def initFile(self):
		fn = os.listdir(self.data_dir)
		self.img_files = sorted(fn, key=lambda x:int(x[:x.index('.')]) )

	def norm201(self, x):
		x_min = x.reshape(-1, 3).min(axis=0)
		x_max = x.reshape(-1, 3).max(axis=0)
		x = (x - x.min()) / (x.max() - x.min())
		return x

	def __getitem__(self, index):

		fn = self.img_files[index]
		# x = Image.open(self.data_dir + '/img/' + fn).convert('RGB')
		x = cv2.cvtColor(cv2.imread(self.data_dir + fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(float)
		norm_x = self.norm201(x)
		x_ten = torch.FloatTensor(norm_x).permute((2,0,1))
		x_ten = trns.Normalize(
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x_ten)

		return x_ten

	def __len__(self):
		return len(self.img_files)



def predict(model, valid_data):

	model = model.cuda()
	model = model.eval()

	preds = []

	with torch.no_grad():
		for img in valid_data:
			batch_size = img.shape[0]

			img = img.cuda()
			out = model(img)
			pred = out.argmax(dim=1)

			preds.append(pred.cpu().numpy())

	preds = np.concatenate(preds, axis=0)
	
	model = model.cpu()

	return preds




if __name__ == '__main__':


	parser = ArgumentParser()
	parser.add_argument("source_dir")
	parser.add_argument("pred_dir")
	parser.add_argument("model_path")

	args = parser.parse_args()

	dataset_transform = trns.Compose([
			# trns.RandomRotation([0, 360]),
			# trns.ToTensor(),
			# trns.Normalize(mean=[], std=[])
		])

	test_dataset = ImageDataset(args.source_dir + '/', dataset_transform)
	test_dataloader = DataLoader(dataset=test_dataset,
								batch_size=32,
								shuffle=False,
								num_workers=4
					)

	model = ResUNet(3, 9)
	model.load_state_dict(torch.load(args.model_path))

	preds = predict(model, test_dataloader)

	for idx, fn in enumerate(test_dataset.img_files):
		img = Image.fromarray(preds[idx].astype(np.uint8))
		img.save(args.pred_dir + '/' + fn)
