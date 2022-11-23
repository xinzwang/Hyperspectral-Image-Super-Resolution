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




class MultiDataset(Dataset):
	def __init__(self, path, scale_factor=2, test_flag=False):
		super(MultiDataset, self).__init__()
		self.scale_factor = scale_factor
		self.ratio = 1.0/scale_factor
		self.test_flag = test_flag
		self.paths = glob.glob(path + '*.npy')
		data0 = np.load(self.paths[0])
		self.channels = data0.shape[-1]

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		# print(path)
		hr = np.load(path).astype(np.float32)	# HWC; [0, 1]; np.float32

		H, W, C = hr.shape
		if (H % self.scale_factor !=0) or (W % self.scale_factor != 0):
			hr = hr[0:H//self.scale_factor*self.scale_factor, 0:W//self.scale_factor*self.scale_factor, :]

		if self.test_flag:
			center_H = H // 2
			center_W = W // 2
			hr = hr[center_H-256:center_H+256, center_W-256:center_W+256, ...]
		else:
			# enhance
			hr = rot90(hr)
			hr = flip(hr)
			hr = np.ascontiguousarray(hr)
		# down sample
		lr = down_sample(hr, scale_factor=self.scale_factor, kernel_size=(9,9), sigma=3)

		# cv2.imwrite('lr.png', np.mean(lr, axis=2)*255)
		# cv2.imwrite('hr.png', np.mean(hr, axis=2)*255)

		return lr.transpose(2,0,1), hr.transpose(2,0,1)

	if __name__=='__main__':
		data = np.load('/data2/wangxinzhe/codes/datasets/ICVL/test/selfie_0822-0906.npy')
		print(data.dtype)
		print(data.shape)
		print(data.max())
		print(data.min())

		lr = down_sample(data, scale_factor=2, kernel_size=(9,9), sigma=3)

		cv2.imwrite('lr.png', np.mean(lr, axis=2)*255)
		cv2.imwrite('hr.png', np.mean(data, axis=2)*255)


