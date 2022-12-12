"""
SingleDataset

One file with all training images. 

Data file must be stored in the form of numpy.array, with the shape of [N, H, W, C].
Where N represents the number of image, H and W represent the image width, and C represents the number of channels
"""

import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SingleDataset(Dataset):
	def __init__(self, path, scale_factor=2, test_flag=False, p=0.1):
		super(SingleDataset, self).__init__()

		self.scale_factor = scale_factor
		self.test_flag = test_flag

		print('[SingleDataset %s] load dataset' % ('TEST' if test_flag else 'TRAIN'))
		self.lr = np.load(path[0])	# NHWC; [0, 1]; float32
		self.hr = np.load(path[1])	# NHWC; [0, 1]; float32

		# use P percent train data
		if not test_flag:
			l = self.lr.shape[0]
			self.lr = self.lr[0:int(l*p), ...]
			self.hr = self.hr[0:int(l*p), ...]
			print('Before:%d After:%d' %(l, self.lr.shape[0]))
			
		self.shape = self.hr.shape
		self.channels = self.shape[-1]
		self.len = self.shape[0]
		return
	
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		hr = self.hr[index, ...]
		lr = self.lr[index, ...]

		lr = np.ascontiguousarray(lr.transpose(2, 0, 1))
		hr = np.ascontiguousarray(hr.transpose(2, 0, 1))

		lr = torch.from_numpy(lr).float()
		hr = torch.from_numpy(hr).float()

		# lms = F.interpolate(lr.unsqueeze(0),  scale_factor=self.scale_factor, mode='bicubic', align_corners=False).squeeze(0)
		# return lr, lms, hr
		
		return lr, hr

