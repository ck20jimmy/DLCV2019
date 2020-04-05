import numpy as np
import sys
import os
from PIL import Image
import cv2

from tqdm import tqdm

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

from tensorboardX import SummaryWriter

from data import ImageDataset
from model import *
from train import *

from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

import os
import cv2
import torch
import numpy as np


class PredictImageDataset(Dataset):


	def __init__(self, data_dir):
        
		self.data_dir = data_dir
		self.img_files = None
		self.initFile()

	def initFile(self):
		fn = os.listdir(self.data_dir)
		fn = [ f for f in fn if f.split('.')[-1] == 'png' ]
		self.img_files = fn

	def normalize(self, x):
		out = (x / 255.0) * 2 - 1
		return out


	def __getitem__(self, index):

		fn = self.img_files[index]
		img = cv2.imread(self.data_dir + fn, cv2.IMREAD_COLOR)

		x = img[:,:,0].astype(np.float64)
		x = x.reshape(1, 64, 64)
		norm_x = self.normalize(x)

		x_ten = torch.FloatTensor(norm_x).view(1, 64, 64)

		return x_ten

		
	def __len__(self):
		return len(self.img_files)


def model_predict(model, inp_img, save_dir):

	model = model.eval()

	with torch.no_grad():
		
		fake_image = model.generate(inp_img).detach()
		
		fake_image = (fake_image + 1) / 2 * 255.0

		l_image = (inp_img + 1) / 2 * 255.0

		lab_image = torch.cat((l_image, fake_image), dim=1).cpu().numpy().transpose((0,2,3,1))

		for i in range(lab_image.shape[0]):
			lab_image[i] = cv2.cvtColor(lab_image[i].astype(np.uint8), cv2.COLOR_LAB2RGB)

		out_image = torch.from_numpy( lab_image.transpose(0, 3, 1, 2) )

		fn = save_dir + 'all.png'
		save_image(out_image, fn, normalize=True)

		# for i in range(out_image.shape[0]):
			# fn = save_dir + "color_" + str(i) + ".png"
			# save_image(out_image[i], fn, nrow=8, normalize=True)



# https://github.com/ImagingLab/Colorizing-with-GANs
# https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/

if __name__ == '__main__':


	train_data = PredictImageDataset("../../predict/DCGAN_2/")
	train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)

	model = ColorModel()

	model.load_state_dict(torch.load("../../model/ColorModel2/model_100"))

	model = model.cuda()

	save_dir = "../../predict/ColorModel_2/"

	for data in train_dataloader:
		data = data.cuda()
		model_predict(model, data, save_dir)
		data = data.cpu()
		break

	model = model.cpu()