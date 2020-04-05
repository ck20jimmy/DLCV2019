import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import sys
import os
import cv2

from model import *
from train import *


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models

MAX_LENGTH = 20

class FullLengthDataset(Dataset):

    def __init__(self, video_dir):

        self.video_dir = video_dir

        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 3)

        self.prepareImage()

    def prepareImage(self):

        all_image_file = []

        image_fn = [ f for f in os.listdir(self.video_dir) ]
        image_fn = [ os.path.join(self.video_dir, f) \
            for f in sorted(image_fn, key=lambda x:int(x.split('.')[0])) ]
        
        for i in range( len(image_fn) // MAX_LENGTH + 1 ):
            all_image_file.append(image_fn[ i*MAX_LENGTH:(i+1)*MAX_LENGTH ])

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

        mask = np.zeros(MAX_LENGTH).astype(int)
        mask[:video.shape[0]] = 1

        pad_emb = np.zeros( (MAX_LENGTH - video_len, 3, video.shape[2], video.shape[3]) )
        video = np.concatenate((video, pad_emb), axis=0)

        pad_emb = np.zeros(MAX_LENGTH - video_len)

        return video, mask

    def __len__(self):
        return len(self.image_file)



def collate_fn(data):

    video, video_mask = zip(*data)
    
    video = np.stack(video)
    video_mask = np.stack(video_mask)

    video_mask_ten = torch.from_numpy(video_mask).bool()
    video_ten = torch.from_numpy(video).float()

    return video_ten, video_mask_ten



if __name__ == '__main__':

    
    test_video_dir = sys.argv[1] + '/'

    out_dir = sys.argv[2] + '/'
    model_path = sys.argv[3]

    pretrain = models.resnet50(pretrained=True)
    model = Model(pretrain)

    model.load_state_dict(torch.load(model_path))

    model = model.cuda()

    for task in os.listdir(test_video_dir):
        valid_dataset = FullLengthDataset( os.path.join(test_video_dir, task) )
        valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8, num_workers=3, collate_fn=collate_fn)

        pred = predict(model, valid_dataloader)

        with open( os.path.join(out_dir, task + '.txt'), 'w') as fd:
            for i in range(pred.shape[0]):
                if i != 0:
                    fd.write('\n')
                fd.write(str(pred[i]))
            fd.close()
        

        