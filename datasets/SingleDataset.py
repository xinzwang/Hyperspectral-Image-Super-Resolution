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
from torch.utils.data import Dataset

class SingleDataset(Dataset):
	def __init__(self, path, scale_factor=2, test_flag=False):
		super(SingleDataset, self).__init__()

		self.scale_factor = scale_factor
		self.test_flag = test_flag

		self.lr = np.load(path[0])	# NHWC; [0, 1]; float32
		self.hr = np.load(path[1])	# NHWC; [0, 1]; float32

		self.shape = self.hr.shape
		self.channels = self.shape[-1]
		self.len = self.shape[0]
		return
	
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		hr = self.hr[index, ...]
		lr = self.lr[index, ...]

		return lr.transpose(2,0,1), hr.transpose(2,0,1)


if __name__ == '__main__':
	import glob
	import cv2
	import numpy as np

	paths = glob.glob('/data2/wangxinzhe/codes/datasets/CAVE/hsi/*.npy')
	print(paths)
	for i, x in enumerate(paths):
		gt = np.load(x)
		print('gt shape', gt.shape)
		print('gt min:', gt.min())
		print('gt max:', gt.max())

		cv2.imwrite('img/cave/%s.png'%(x.split('/')[-1].split('.')[0]), np.mean(gt, axis=2)/ gt.max() * 255 )
