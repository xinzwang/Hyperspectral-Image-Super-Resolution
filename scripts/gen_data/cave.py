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


import os
import argparse
import glob
import numpy as np
from common import gen_panel_from_area


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='/data2/wangxinzhe/codes/datasets/CAVE/hsi/')
	parser.add_argument('--save_path', default='../../data/CAVE/')
	parser.add_argument('--train_num', default=20)
	parser.add_argument('--val_num', default=3)
	parser.add_argument('--patch_size', default=64)
	
	args = parser.parse_args()
	print(args)
	return args

def run(args):
	paths = glob.glob(args.data_path + '*.npy')
	assert len(paths) == 32, Exception('CAVE should have 32 scenes')

	np.random.shuffle(paths)

	# train datas
	train_panel = []
	for path in paths[0:args.train_num]:
		data = np.load(path)
		print(data.shape, data.max(), data.min())
		train_panel += gen_panel_from_area(data, panel_size=args.patch_size, rot=0)
		train_panel += gen_panel_from_area(data, panel_size=args.patch_size, rot=1)	# rotate
		train_panel += gen_panel_from_area(data, panel_size=args.patch_size, rot=2)	# rotate
		train_panel += gen_panel_from_area(data, panel_size=args.patch_size, rot=3)	# rotate


	train_panel = np.array(train_panel)
	print('train shape:', train_panel.shape)
	print('train max:', train_panel.max())
	print('train min:', train_panel.min())

	# eval datas
	val_img = []
	for path in paths[args.train_num:args.train_num + args.val_num]:
		data = np.load(path)
		val_img.append(data)
	val_img = np.array(val_img)
	print('val shape:', val_img.shape)
	print('val max:', val_img.max())
	print('val min:', val_img.min())


	# test datas
	test_img = []
	for path in paths[args.train_num + args.val_num:]:
		data = np.load(path)
		test_img.append(data)
	test_img = np.array(test_img)
	print('test shape:', test_img.shape)
	print('test max:', test_img.max())
	print('test min:', test_img.min())

	# save data
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	np.save(args.save_path + 'train.npy', train_panel)
	np.save(args.save_path + 'val.npy', val_img)
	np.save(args.save_path + 'test.npy', test_img)


if __name__=='__main__':
	args = parse_args()
	run(args)