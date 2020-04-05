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

from torchvision.utils import save_image
from color_model import ColorModel


def model_predict(model, inp_img, fn):

	model = model.eval()

	with torch.no_grad():
		
		fake_image = model.generate(inp_img).detach()
		
		fake_image = (fake_image + 1) / 2 * 255.0

		l_image = (inp_img + 1) / 2 * 255.0

		lab_image = torch.cat((l_image, fake_image), dim=1).cpu().numpy().transpose((0,2,3,1))

		for i in range(lab_image.shape[0]):
			lab_image[i] = cv2.cvtColor(lab_image[i].astype(np.uint8), cv2.COLOR_LAB2RGB)

		out_image = torch.from_numpy( lab_image.transpose(0, 3, 1, 2) )

		save_image(out_image, fn, normalize=True, nrow=4)




if __name__ == '__main__':


	out_fn = sys.argv[1] + '/' + "fig_1_2.jpg"

	model = ContourGAN(NOISE_SIZE, IMAGE_SIZE, INPUT_CHANNEL)
	model.load_state_dict(torch.load("./model/DCGAN"))

	Color_model = ColorModel()
	Color_model.load_state_dict(torch.load("./model/ColorModel"))

	# fixed_noise = torch.randn(64, NOISE_SIZE, 1, 1)
	# fixed_noise = fixed_noise.reshape(64, NOISE_SIZE, 1, 1)

	fixed_noise = torch.from_numpy(np.load("./model/DCGAN_noise.npy")).float()


	model = model.cuda()
	
	with torch.no_grad():
		fixed_noise = fixed_noise.cuda()
		fake_image = model.generate(fixed_noise).detach().cpu()
		model_predict(Color_model, fake_image, out_fn)

	model = model.cpu()


