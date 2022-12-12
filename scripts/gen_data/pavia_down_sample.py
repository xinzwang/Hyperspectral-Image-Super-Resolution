"""
CAVE dataset down sample

"""


import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm

sys.path.append("../..") 
from datasets.common import Downsample


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='../../data/Pavia/')
	parser.add_argument('--save_path', default='../../data/Pavia/bicubic+gauss/')
	
	args = parser.parse_args()
	print(args)
	return args

def run(args):
	train_data = np.load(args.data_path + 'train.npy')
	test_data = np.load(args.data_path + 'test.npy')

	# cal band mean in train data
	band_mean = np.mean(train_data, axis=(0, 1, 2))
	print(band_mean)
	print(train_data.shape)

	# down sample
	for scale in [2, 4, 8]:
		# train
		down_train = []
		for img in tqdm(train_data):
			# img = down_sample(img, scale, kernel_size=(9, 9), sigma=3)
			# img = bicubic_downsample(img, scale)
			img = Downsample(img, scale, sigma=3, ksize=9)
			down_train.append(img)

		# test
		down_test = []
		for img in tqdm(test_data):
			# img = down_sample(img, scale, kernel_size=(9, 9), sigma=3)
			# img = bicubic_downsample(img, scale)
			img = Downsample(img, scale, sigma=3, ksize=9)
			down_test.append(img)

		down_train = np.array(down_train)
		down_test = np.array(down_test)

		# save result
		print(down_train.shape)
		print(down_test.shape)
		np.save(args.save_path + 'train_scale=%d.npy' %(scale), down_train)
		np.save(args.save_path + 'test_scale=%d.npy' %(scale), down_test)

if __name__=='__main__':
	args = parse_args()
	run(args)