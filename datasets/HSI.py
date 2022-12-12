import os
import math
import glob
import numpy as np
import scipy.io as scio
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if __name__=='__main__':
	from common import Augment
else:
	from .common import Augment


class HSIDataset(Dataset):
	def __init__(self, lr_path, hr_path, lms_path=None, test_flag=False, cache_ram=False, use_lms=False):
		super().__init__()

		self.test_flag = test_flag
		self.cache_ram = cache_ram
		self.use_lms = use_lms

		print('[HSIDataset] %s' % ('TEST' if test_flag else 'TRAIN'))

		# Data file paths
		self.lr_paths = sorted(glob.glob(lr_path + '*.mat'))
		self.hr_paths = sorted(glob.glob(hr_path + '*.mat'))
		if use_lms:
			self.lms_paths = sorted(glob.glob(lms_path + '*.mat'))
		print('[HSIDataset] hr_num:%d lr_num:%d lms_num:%d' %(len(self.hr_paths), len(self.lr_paths), len(self.lms_paths) if use_lms else -1))


		# Cache data
		if cache_ram:
			self.lr_cache = [scio.loadmat(p)['lr'] for p in tqdm(self.lr_paths)]
			self.hr_cache = [scio.loadmat(p)['hr'] for p in tqdm(self.hr_paths)]
			if use_lms:
				self.lms_cache = [scio.loadmat(p)['lms'] for p in self.lms_paths]
	
	def __len__(self):
		return len(self.lr_paths)

	def __getitem__(self, index):
		if self.cache_ram:
			lr = self.lr_cache[index]
			hr = self.hr_cache[index]
			lms = self.lms_cache[index] if self.use_lms else None
		else:
			lr = scio.loadmat(self.lr_paths[index])['lr']
			hr = scio.loadmat(self.hr_paths[index])['hr']
			lms = scio.loadmat(self.lms_paths[index])['lms'] if self.use_lms else None

		if not self.test_flag:
			lr, hr, lms = Augment([lr, hr ,lms])

		lr = np.ascontiguousarray(lr.transpose(2, 0, 1))
		hr = np.ascontiguousarray(hr.transpose(2, 0, 1))
		
		# [0, 255]->[0, 1]
		lr = torch.from_numpy(lr).float()
		hr = torch.from_numpy(hr).float()

		if self.use_lms:
			lms = np.ascontiguousarray(lms.transpose(2, 0, 1))
			lms = torch.from_numpy(lms).float()
			return lr, lms, hr
		else:
			return lr, hr


if __name__=='__main__':
	import cv2

	dataset = HSIDataset(
		hr_path ='/data2/wangxinzhe/codes/datasets/CAVE/train/HR/',
		lr_path ='/data2/wangxinzhe/codes/datasets/CAVE/train/LR/X4/',
		lms_path='/data2/wangxinzhe/codes/datasets/CAVE/train/LMS/X4/',
		test_flag=True,
		cache_ram=False,
		use_lms=True)

	band_mean = []
	for i in range(dataset.__len__()):
		lr, hr, lms =  dataset.__getitem__(i)


		hr = hr.cpu().numpy().transpose(1,2,0)

		band_mean.append(hr.mean(axis=(0,1)))

	print(np.mean(band_mean, axis=0))
