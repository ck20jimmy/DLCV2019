import numpy as np
import sys
import os
from PIL import Image
import cv2
import pickle

# import tqdm

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns

from tensorboardX import SummaryWriter


class BaselineModel(nn.Module):

	def __init__(self):
		super(BaselineModel, self).__init__()

		self.resnet18 = models.resnet18(pretrained=True)
		
		self.embedding = nn.Sequential(
			self.resnet18.conv1,
	        self.resnet18.bn1,
	        self.resnet18.relu,
	        self.resnet18.maxpool,

	        self.resnet18.layer1,
	        self.resnet18.layer2,
	        self.resnet18.layer3,
	        self.resnet18.layer4,
			)

		self.model = nn.Sequential(
				nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
				nn.ReLU(),

				nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
				nn.ReLU(),

				nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
				nn.ReLU(),

				nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
				nn.ReLU(),

				nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
				nn.ReLU(),

				nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)
			)

	def forward(self, inp):
		emb = self.embedding(inp)
		out = self.model(emb)

		return out



class ImageDataset(Dataset):


	def __init__(self, data_dir, transform):
        
		self.data_dir = data_dir
		self.img_files = None
		self.initFile()

		self.transform = transform

	def initFile(self):
		fn = os.listdir(self.data_dir + '/img/')
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
		
		# normalize to 0-1
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

    return overlap, union



def train(model, optim, train_data, valid_data):

	
	criterion = nn.CrossEntropyLoss()

	# training
	model = model.train()
	train_loss = 0.0
	train_iters = 0.0

	all_train_loss = []
	
	train_overlap = torch.zeros(9).cuda()
	train_union = torch.zeros(9).cuda()

	for img, label in train_data:

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

		train_loss += loss
		all_train_loss.append(loss.item())

		train_iters += 1

		overlap, union = mean_iou_score(pred, label)
		train_overlap += overlap
		train_union += union

		

	train_loss /= train_iters
	train_miou  = (train_overlap / train_union).mean()


	# evaluation
	model = model.eval()
	valid_loss = 0.0
	valid_iters = 0.0

	valid_overlap = torch.zeros(9).cuda()
	valid_union = torch.zeros(9).cuda()

	with torch.no_grad():
		for img, label in valid_data:
			batch_size = img.shape[0]

			img = img.cuda()
			label = label.cuda()

			out = model(img)
			pred = out.argmax(dim=1)

			target = label.resize(batch_size, 352*448)
			loss = criterion(out.resize(batch_size, 9, 352*448), target)
			
			valid_loss += loss
			valid_iters += 1

			overlap, union = mean_iou_score(pred, label)
			valid_overlap += overlap
			valid_union += union

	valid_loss /= valid_iters
	valid_miou = (valid_overlap / valid_union).cpu().numpy() #.mean()

	# print("Train mIoU:{:.6}\tValid mIoU:{:.6}".format(train_miou.item(), valid_miou.item()))

	return train_loss.item(), valid_loss.item(), train_miou.item(), valid_miou, all_train_loss





if __name__ == '__main__':


	dataset_transform = trns.Compose([
			# trns.RandomRotation([0, 360]),
			# trns.RandomHorizontalFlip(),
			# trns.ToTensor(),
			# trns.Normalize(mean=[], std=[])
		])

	train_dataset = ImageDataset("./aug_data/train/", dataset_transform)
	valid_dataset = ImageDataset("./aug_data/val/", dataset_transform)

	train_dataloader = DataLoader(dataset=train_dataset,
									batch_size=32,
									shuffle=True,
									num_workers=4
						)
	
	valid_dataloader = DataLoader(dataset=valid_dataset,
								batch_size=128,
								shuffle=False,
								num_workers=4
					)


	model = BaselineModel()
	optimizer = torch.optim.Adam( model.parameters(), lr=5e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
		 factor=0.5, patience=5, verbose=True)



	epochs = 50
	model = model.cuda()

	best_valid_miou = -1

	out_dir = "./model/SimpleModel/"

	writer = SummaryWriter(logdir=out_dir)

	all_loss = []
	all_valid_iou = []

	for ep in range(epochs):

		train_loss, valid_loss, train_miou, valid_iou, all_train_loss = \
				train(model, optimizer, train_dataloader, valid_dataloader)

		valid_miou = valid_iou.mean()

		print("Epoch:{}\tTrain Loss:{:.6}\tValid Loss:{:.6}".format(ep+1, train_loss, valid_loss))
		print("Epoch:{}\tTrain mIoU:{:.6}\tValid mIoU:{:.6}".format(ep+1, train_miou, valid_miou))
		print("===========================================================")

		all_loss += all_train_loss
		all_valid_iou.append(valid_iou)

		if valid_miou > best_valid_miou:
			best_valid_miou = valid_miou
			torch.save(model.state_dict(), out_dir + "baseline_model")

		for param_group in optimizer.param_groups:
			current_lr = param_group['lr']

		scheduler.step(valid_loss)

		writer.add_scalar("loss/train_epoch_loss", train_loss, ep+1)
		writer.add_scalar("loss/valid_epoch_loss", valid_loss, ep+1)
		writer.add_scalar("mIoU/train_epoch_mIoU", train_miou, ep+1)
		writer.add_scalar("mIoU/valid_mIoU", valid_miou, ep+1)
		writer.add_scalar("learning_rate", current_lr, ep+1)

		
	all_loss = np.array(all_loss)
	np.save(out_dir+'/training_loss', all_loss)

	all_valid_iou = np.stack(all_valid_iou)
	np.save(out_dir+'/valid_iou', all_valid_iou)

