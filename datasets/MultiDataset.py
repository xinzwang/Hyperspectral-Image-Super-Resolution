"""
MultiDataset

Images in one folder

Some dataset may be too big to load into RAM. MultiDataset only load one image when generate train data.
Data file must be stored in the form of numpy.array, with the shape of [H, W, C].
"""

import os
import math
import numpy as np
import torch
import glob
from pathlib import Path
from torch.utils.data import Dataset

import torchvision.transforms as transforms

if __name__=='__main__':
	from common import CenterCrop, Augment, Downsample
else:
	from .common import CenterCrop, Augment, Downsample


class MultiDataset(Dataset):
	def __init__(self, path, scale_factor=2, test_flag=False):
		super(MultiDataset, self).__init__()
		self.scale_factor = scale_factor
		self.test_flag = test_flag
		self.paths = glob.glob(path + '*.npy')

		data0 = np.load(self.paths[0])
		self.channels = data0.shape[-1]

		self.trans = transforms.Compose(
			[transforms.ToTensor()]
		)

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		hr = np.load(self.paths[index]).astype(np.float32)	# HWC; [0, 1]; np.float32

		if not self.test_flag:
			hr = Augment(hr, hflip=True, rot=True)[0]
			hr = np.ascontiguousarray(hr)

		lr = self.trans(Downsample(hr, self.scale_factor, sigma=3, ksize=9))
		hr = self.trans(hr)

		return lr, hr


if __name__=='__main__':
	import cv2
	train_path = '/data2/wangxinzhe/codes/datasets/ICVL/train/'
	test_path  = '/data2/wangxinzhe/codes/datasets/ICVL/val/'

	train_set = MultiDataset(train_path, scale_factor=4, test_flag=False)
	test_set = MultiDataset(test_path, scale_factor=4, test_flag=True)

	print('Start')

	for i in range(0, 3):
		lr, hr = train_set.__getitem__(i)
		cv2.imwrite('img/train_%d_lr.png'%(i), np.mean(lr.numpy(), axis=0)*255)
		cv2.imwrite('img/train_%d_hr.png'%(i), np.mean(hr.numpy(), axis=0)*255)

	
	for i in range(0, 3):
		lr, hr = test_set.__getitem__(i)
		cv2.imwrite('img/test_%d_lr.png'%(i), np.mean(lr.numpy(), axis=0)*255)
		cv2.imwrite('img/test_%d_hr.png'%(i), np.mean(hr.numpy(), axis=0)*255)



