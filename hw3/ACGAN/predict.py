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

		save_image(out_image, fn, normalize=True, nrow=10)



if __name__ == '__main__':


	out_fn = sys.argv[1] + '/' + "fig2_2.jpg"

	Contour_model = ACGAN()
	Contour_model.load_state_dict(torch.load("./model/ACGAN"))

	Color_model = ColorModel()
	Color_model.load_state_dict(torch.load("./model/ColorModel"))

	fixed_inp = torch.zeros(20, 127)
	fixed_noise = torch.from_numpy(np.load("./model/ACGAN_noise.npy")).float()
	fixed_noise_smile = torch.zeros(20).view(20, 1)
	for i in range(10):
		fixed_inp[2*i] = fixed_noise[i]
		fixed_inp[2*i+1] = fixed_noise[i]

		fixed_noise_smile[2*i] = 0
		fixed_noise_smile[2*i+1] = 1


	Contour_model = Contour_model.cuda()
	Color_model = Color_model.cuda()

	with torch.no_grad():
		fixed_inp = fixed_inp.cuda()
		fixed_noise_smile = fixed_noise_smile.cuda()
		fake_image = Contour_model.generate(fixed_inp, fixed_noise_smile).detach()
		model_predict(Color_model, fake_image, out_fn)

	Contour_model = Contour_model.cpu()
	Color_model = Color_model.cpu()
	
