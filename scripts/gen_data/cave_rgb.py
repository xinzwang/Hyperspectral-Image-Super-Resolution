"""
Generate CAVE dataset

band mean:
	[0.09605248 0.08241889 0.09570218 0.09395839 0.09162653 0.09839191
 	 0.10027285 0.09311603 0.10635328 0.09380119 0.0900033  0.08940635
 	 0.07584481 0.09062859 0.09864467 0.09499688 0.09360316 0.08809625
 	 0.09461606 0.09572747 0.09203358 0.09125434 0.09688216 0.08512419
 	 0.08136569 0.09667883 0.09496876 0.09517377 0.09669999 0.10196626
 	 0.0968043 ]
"""


import cv2
import os
import argparse
import glob
import numpy as np
from common import gen_panel_from_area


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='/data2/wangxinzhe/codes/datasets/CAVE/raw/')
	parser.add_argument('--save_path', default='../../data/CAVE/')
	parser.add_argument('--train_num', default=20)
	parser.add_argument('--val_num', default=3)
	parser.add_argument('--patch_size', default=64)
	
	args = parser.parse_args()
	print(args)
	return args

def collect_data(path):
	dir_path = glob.glob(path + '*_ms')
	print('file_num:', len(dir_path))
	assert len(dir_path) == 32, Exception('CAVE should have 32 scenes')
	
	res = []
	for p in dir_path:
		name = p.split('/')[-1]
		band_imgs = glob.glob(p + '/' + name + '/*.png')

		hsi = []
		for band_path in band_imgs:
			if name == 'watercolors_ms':
				band = cv2.imread(band_path, cv2.IMREAD_GRAYSCALE) / 255
			else:
				band= cv2.imread(band_path, -1) / 65535
			hsi.append(band)
		hsi = np.array(hsi)
		print(f'[Collect] {name} shape:{hsi.shape} max:{hsi.max()} min:{hsi.min()} dtype:{hsi.dtype}')
		res.append(hsi.transpose(1,2,0))
	
	return np.array(res).astype(np.float32)


def run(args):
	# Load cave images
	cave_images = collect_data(args.data_path)	# [H, W, C]; [0, 1]; np.float32

	# Shuffle all images
	np.random.shuffle(cave_images)

	# Generate train val test images
	train_images = cave_images[0:args.train_num]
	val_images = cave_images[args.train_num:args.train_num + args.val_num]
	test_images = cave_images[args.train_num + args.val_num:]
	print(f'[Gen set] train_num:{len(train_images)} val_num:{len(val_images)} test_num:{len(test_images)}')

	# Generate train panels
	train_patches = []
	for img in train_images:
		train_patches  += gen_panel_from_area(img, panel_size=args.patch_size, rot=0)
		# train_patches += gen_panel_from_area(img, panel_size=args.patch_size, rot=1)	# rotate
		# train_patches += gen_panel_from_area(img, panel_size=args.patch_size, rot=2)	# rotate
		# train_patches += gen_panel_from_area(img, panel_size=args.patch_size, rot=3)	# rotate
	train_patches = np.array(train_patches)

	# Band mean
	band_mean = np.mean(train_patches, axis=(0, 1, 2))
	print(f'[Band mean] {band_mean}')

	print(f'[Train] {train_patches.shape}')
	print(f'[Val]   {val_images.shape}')
	print(f'[Tesl]  {test_images.shape}')

	# Save dataset
	np.save(args.save_path + 'train.npy', train_patches)
	np.save(args.save_path + 'val.npy', val_images)
	np.save(args.save_path + 'test.npy', test_images)



if __name__=='__main__':
	args = parse_args()
	run(args)