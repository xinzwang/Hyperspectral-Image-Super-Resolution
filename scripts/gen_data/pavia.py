"""
Generate Pavia dataset
"""


import cv2
import os
import argparse
import glob
import numpy as np
from common import gen_panel_from_area

from scipy.io import loadmat


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='/data2/wangxinzhe/codes/datasets/Pavia/raw/Pavia.mat')
	parser.add_argument('--save_path', default='../../data/Pavia/')
	# parser.add_argument('--test_area', default= ((192, 45), (320, 173)))
	parser.add_argument('--test_area', default=((420, 230), (676, 486)))	# Pavia
	parser.add_argument('--patch_size', default=64)

	
	args = parser.parse_args()
	print(args)
	return args


def run(args):
	pavia_image = loadmat(args.data_path)['pavia']

	# norm
	max_ = pavia_image.max()
	min_ = pavia_image.min()
	pavia_image = (pavia_image - min_) / (max_ - min_)
	pavia_image = pavia_image.astype(np.float32)

	# Generate train test area
	test_area =args.test_area

	train_area1 = pavia_image[:test_area[0][0], ...]
	train_area2 = pavia_image[test_area[0][0]:test_area[1][0], :test_area[0][1], ...]
	train_area3 = pavia_image[test_area[0][0]:test_area[1][0], test_area[1][1]:, ...]
	train_area4 = pavia_image[test_area[1][0]:, ...]

	test_area = pavia_image[test_area[0][0]:test_area[1][0], test_area[0][1]:test_area[1][1], ...]
	test_area = np.array([test_area])
	print('Test area:', test_area.shape)

	# Generate train patches
	train_patches = []
	train_patches += gen_panel_from_area(train_area1, panel_size=64, rot=0)
	train_patches += gen_panel_from_area(train_area2, panel_size=64, rot=0)
	train_patches += gen_panel_from_area(train_area3, panel_size=64, rot=0)
	train_patches += gen_panel_from_area(train_area4, panel_size=64, rot=0)

	train_patches += gen_panel_from_area(train_area1, panel_size=64, rot=1)
	train_patches += gen_panel_from_area(train_area2, panel_size=64, rot=1)
	train_patches += gen_panel_from_area(train_area3, panel_size=64, rot=1)
	train_patches += gen_panel_from_area(train_area4, panel_size=64, rot=1)

	train_patches += gen_panel_from_area(train_area1, panel_size=64, rot=2)
	train_patches += gen_panel_from_area(train_area2, panel_size=64, rot=2)
	train_patches += gen_panel_from_area(train_area3, panel_size=64, rot=2)
	train_patches += gen_panel_from_area(train_area4, panel_size=64, rot=2)

	train_patches += gen_panel_from_area(train_area1, panel_size=64, rot=3)
	train_patches += gen_panel_from_area(train_area2, panel_size=64, rot=3)
	train_patches += gen_panel_from_area(train_area3, panel_size=64, rot=3)
	train_patches += gen_panel_from_area(train_area4, panel_size=64, rot=3)

	train_patches = np.array(train_patches)

	# Band mean
	band_mean = np.mean(train_patches, axis=(0, 1, 2))
	print(f'[Band mean] {band_mean}')

	print(f'[Train] {train_patches.shape}')
	print(f'[Test]  {test_area.shape}')

	# Save dataset
	np.save(args.save_path + 'train.npy', train_patches)
	np.save(args.save_path + 'test.npy', test_area)



if __name__=='__main__':
	args = parse_args()
	run(args)