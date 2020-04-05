import numpy as np
import sys
import os
from PIL import Image
import cv2

from tqdm import tqdm_notebook as tqdm

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

import matplotlib.pyplot as plt



def model_evaluate(feat_extracter, source_classifier, test_dataloader):

    feat_extracter = feat_extracter.eval()
    source_classifier = source_classifier.eval()

    trange = tqdm(test_dataloader)

    all_class = []
    
    with torch.no_grad():
        
        for data in trange:

            inp_img, inp_img_norm = data

            inp_img = inp_img.cuda()

            test_feat = feat_extracter(inp_img)
            class_prob = source_classifier(test_feat)

            class_pred = class_prob.argmax(dim=1).view(-1, 1)
            all_class.append(class_pred.cpu().numpy())

    return all_class




if __name__ == '__main__':


    data_dir = sys.argv[1] + "/"
    out_fn = sys.argv[2]

    transform = trns.Compose([])

    svhn_testdata = ImageDataset(data_dir, "", transform )
    svhn_test_dataloader = DataLoader(svhn_testdata, batch_size=128, shuffle=False, num_workers=1)

    model = GTA()
    model.load_state_dict(torch.load("./model/GTA_M2S"))

    model = model.cuda()
    svhn_class = model_evaluate(model.feat_extracter, model.source_classifier, svhn_test_dataloader)
    model = model.cpu()

    svhn_class = np.concatenate(svhn_class)

    with open(out_fn, 'w') as fd:
    	for i in range(svhn_class.shape[0]):
    		if i != 0:
    			fd.write('\n')

    		fd.write(svhn_testdata.img_files[i])
    		fd.write(',')
    		fd.write(str(svhn_class[i,0]))

    	fd.close()
