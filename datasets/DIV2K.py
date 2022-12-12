import cv2
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from .common import RandomPatch, Augment, Downsample

class DIV2KDataset(Dataset):
	def __init__(self, path, scale_factor, panel_size=64, test_flag=False):
		super().__init__()

		self.panel_size = panel_size
		self.scale_factor = scale_factor
		self.test_flag = test_flag

		self.path_hr = sorted(glob.glob(path[1] + '*.png'))
		self.path_lr = sorted(glob.glob(path[0] + '*.png'))

		# if test_flag:
		# 	# self.path_lr = self.path_lr[0:50]
		# 	self.path_hr = self.path_hr[0:50]
		# else:
		# 	# self.path_lr = self.path_lr[50:]
		# 	self.path_hr = self.path_hr[50:]
		
		
		print('[DIV2K Dataset] read hr data')
		# self.hr = []
		# for p in tqdm(self.path_hr):
		# 	img = cv2.imread(p).astype(np.float32)
		# 	H, W, C = img.shape
		# 	img = img[0:-(H%self.scale_factor), :, :] if H%self.scale_factor != 0 else img
		# 	img = img[:, 0:-(W%self.scale_factor), :] if W%self.scale_factor != 0 else img
		# 	self.hr.append(img)
		# self.lr = [Downsample(img, self.scale_factor, sigma=3, ksize=9) for img in self.hr]
		# print('[DIV2K Dataset] read lr data')
		self.hr = [cv2.imread(p).astype(np.float32) for p in tqdm(self.path_hr)]

		self.lr = [Downsample(img, self.scale_factor, sigma=3, ksize=9) for img in self.hr]
		# self.lr = [np.expand_dims(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY).astype(np.float32), axis=2) for p in tqdm(self.path_lr)]

		self.len = len(self.path_hr)

	def _get_patch(self, lr, hr):
		if not self.test_flag:	# train
			lr, hr = RandomPatch(lr, hr, patch_size=self.panel_size, scale=self.scale_factor, multi_scale=False)
			lr, hr = Augment([lr, hr])	# flip, rot
		else:	# test
			lr = lr[0:512 // self.scale_factor, 0:512 // self.scale_factor, :]
			ih, iw = lr.shape[0:2]
			hr = hr[0:ih * self.scale_factor, 0:iw * self.scale_factor, :]
		return lr, hr
	
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		lr = self.lr[index]
		hr = self.hr[index]
		lr, hr = self._get_patch(lr, hr)

		# HWC->CHW
		lr = np.ascontiguousarray(lr.transpose(2, 0, 1))
		hr = np.ascontiguousarray(hr.transpose(2, 0, 1))
		
		# [0, 255]->[0, 1]
		lr = torch.from_numpy(lr).float() / 255.0
		hr = torch.from_numpy(hr).float() / 255.0

		return lr, hr



