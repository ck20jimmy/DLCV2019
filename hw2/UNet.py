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

from model import UNet, ResUNet


class ImageDataset(Dataset):


	def __init__(self, data_dir, transform):
        
		self.data_dir = data_dir
		self.img_files = None
		self.initFile()

		self.transform = transform

	def initFile(self):
		fn = os.listdir(self.data_dir + '/img/')
		fn = [ f for f in fn if f.split('.')[-1] == 'png' ]
		self.img_files = sorted(fn, key=lambda x:int(x.split('.')[0].split('_')[0]) )

	def norm201(self, x):
		x_min = x.reshape(-1, 3).min(axis=0)
		x_max = x.reshape(-1, 3).max(axis=0)
		x = (x - x.min()) / (x.max() - x.min())
		return x

	def __getitem__(self, index):

		fn = self.img_files[index]
		# x = Image.open(self.data_dir + '/img/' + fn).convert('RGB')
		x = cv2.cvtColor(cv2.imread(self.data_dir + '/img/' + fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(float)
		norm_x = self.norm201(x)
		x_ten = torch.FloatTensor(norm_x).permute((2,0,1))
		
		x_ten = trns.Normalize(
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x_ten)

		y = cv2.imread(self.data_dir + '/seg/' + fn, cv2.IMREAD_GRAYSCALE).astype(float)
		y_ten = torch.LongTensor(y)

		return x_ten, y_ten

	def __len__(self):
		return len(self.img_files)



def mean_iou_score(pred, labels, num_classes=9):
    '''
    Compute mean IoU score over 9 classes
    '''

    overlap = torch.zeros(9).cuda()
    union = torch.zeros(9).cuda()
    
    for i in range(num_classes):
        tp_fp = torch.sum(pred == i)
        tp_fn = torch.sum(labels == i)
        tp = torch.sum((pred == i) * (labels == i))
        
        overlap[i] = tp
        union[i] = (tp_fp + tp_fn - tp)

    overlap = overlap.cpu()
    union = union.cpu()

    return overlap.numpy(), union.numpy()



def train(model, optim, train_data, valid_data):
	
	criterion = nn.CrossEntropyLoss()

	# training
	model = model.train()
	train_loss = 0.0
	train_iters = 0.0

	all_train_loss = []
	
	train_overlap = np.zeros(9) #torch.zeros(9).cuda()
	train_union = np.zeros(9) #torch.zeros(9).cuda()
    
	optim.zero_grad()
    
	for idx, (img, label) in enumerate(tqdm(train_data)):
        
# 		if (idx+1) % 4:
# 			optim.step()
# 			optim.zero_grad()
        
		optim.zero_grad()

		batch_size = img.shape[0]

		img = img.cuda()
		label = label.cuda()

		out = model(img)
		pred = out.argmax(dim=1)

		target = label.resize(batch_size, 352*448)

		loss = criterion(out.resize(batch_size, 9, 352*448), target)
		loss.backward()

		optim.step()

		train_loss += loss.item()
		all_train_loss.append(loss.item())

		train_iters += 1

		overlap, union = mean_iou_score(pred, label)
		train_overlap += overlap
		train_union += union

		img = img.cpu()
		label = label.cpu()


	train_loss /= train_iters
	train_miou  = (train_overlap / train_union).mean()


	# evaluation
	model = model.eval()
	valid_loss = 0.0
	valid_iters = 0.0

	valid_overlap = np.zeros(9) #torch.zeros(9).cuda()
	valid_union = np.zeros(9) #torch.zeros(9).cuda()

	with torch.no_grad():
		for img, label in tqdm(valid_data):
			batch_size = img.shape[0]

			img = img.cuda()
			label = label.cuda()

			out = model(img)
			pred = out.argmax(dim=1)

			target = label.resize(batch_size, 352*448)
			loss = criterion(out.resize(batch_size, 9, 352*448), target)
			
			valid_loss += loss.item()
			valid_iters += 1

			overlap, union = mean_iou_score(pred, label)
			valid_overlap += overlap
			valid_union += union

	valid_loss /= valid_iters
	valid_miou = (valid_overlap / valid_union).mean()


	return train_loss, valid_loss, train_miou, valid_miou, all_train_loss





if __name__ == '__main__':


	dataset_transform = trns.Compose([
			# trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			# trns.ToTensor(),
		])


	train_dataset = ImageDataset("./aug_data/train/", dataset_transform)
	valid_dataset = ImageDataset("./aug_data/val/", dataset_transform)

	train_dataloader = DataLoader(dataset=train_dataset,
									batch_size=16,
									shuffle=True,
									num_workers=7
						)
	
	valid_dataloader = DataLoader(dataset=valid_dataset,
								batch_size=32,
								shuffle=False,
								num_workers=7
					)


	model = ResUNet(3, 9)
	optimizer = torch.optim.Adam( model.parameters(), lr=5e-5, weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
		 factor=0.5, patience=5, verbose=True)

	epochs = 150
	model = model.cuda()

	best_valid_miou = -1

	out_dir = "./model/Unet_34/"

	writer = SummaryWriter(logdir=out_dir)

	all_loss = []
	all_valid_miou = []

	for ep in range(epochs):

		train_loss, valid_loss, train_miou, valid_miou, all_train_loss = \
				train(model, optimizer, train_dataloader, valid_dataloader)

		print("Epoch:{}\tTrain Loss:{:.6}\tValid Loss:{:.6}".format(ep+1, train_loss, valid_loss))
		print("Epoch:{}\tTrain mIoU:{:.6}\tValid mIoU:{:.6}".format(ep+1, train_miou, valid_miou))
		print("===========================================================")

		all_loss += all_train_loss

		if valid_miou > best_valid_miou:
			best_valid_miou = valid_miou
			torch.save(model.state_dict(), out_dir + "improved_model")

		for param_group in optimizer.param_groups:
			current_lr = param_group['lr']

		scheduler.step(valid_loss)

		writer.add_scalar("train_epoch_loss", train_loss, ep+1)
		writer.add_scalar("train_epoch_mIoU", train_miou, ep+1)
		writer.add_scalar("valid_epoch_loss", valid_loss, ep+1)
		writer.add_scalar("valid_mIoU", valid_miou, ep+1)
		writer.add_scalar("learning_rate", current_lr, ep+1)

	import pickle
	with open(out_dir+"/" + 'training_loss', 'wb') as fd:
		pickle.dump(all_loss, fd)
		fd.close()

	writer.close()
