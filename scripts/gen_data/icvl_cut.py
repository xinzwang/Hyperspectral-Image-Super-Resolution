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
from datasets.common import Downsample, CenterCrop


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='/data2/wangxinzhe/codes/datasets/ICVL/val/')
	parser.add_argument('--save_path', default='/data2/wangxinzhe/codes/datasets/ICVL/val_512/')
	
	args = parser.parse_args()
	print(args)
	return args

def run(args):
	paths = glob.glob(args.data_path + '*.npy')
	
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	for p in tqdm(paths):
		name = p.split('/')[-1]
		img = np.load(p)
		img = CenterCrop(img, 512)

		np.save(args.save_path + name, img)


if __name__=='__main__':
	args = parse_args()
	run(args)