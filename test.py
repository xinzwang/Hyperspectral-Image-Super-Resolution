import os
import cv2
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


from utils.tools import build_dataset
from utils.test import test, visual
from utils.core import SRCore

# from models import BiFQRNNREDC3D

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CAVE', choices=['ICVL', 'CAVE', 'Pavia', 'Salinas','PaviaU', 'KSC', 'Indian'])
	parser.add_argument('--scale_factor', default=2, type=int)
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--device', default='cpu')

	parser.add_argument('--ckpt', default='ckpts/CAVE/MCNet/4/2022-11-24_00-42-14/epoch=475_psnr=39.71_ssim=0.98/ckpt.pt')
	# CAVE
	parser.add_argument('--test_hr_path', default='data/CAVE/test.npy')
	parser.add_argument('--test_lr_path', default='data/CAVE/bicubic_gauss_ksize=9_sigma=3/test_scale=4.npy')
	# parser.add_argument('--test_lr_path', default='data/CAVE/bicubic/test_scale=4.npy')

	args = parser.parse_args()
	print(args)
	return args

def run(args):
	device=torch.device(args.device)

	test_dataset, test_dataloader = build_dataset(
		dataset=args.dataset,
		path=(args.test_lr_path, args.test_hr_path), 
		batch_size=16, 
		scale_factor=args.scale_factor, 
		test_flag=True)

	ckpt = torch.load(args.ckpt, map_location=device)
	model = ckpt['model']

	# external_checkpoint = ckpt['model']
	# model = BiFQRNNREDC3D(channels=16, scale_factor=2).to(device)
	# model.load_state_dict(external_checkpoint)


	test(model, test_dataloader, device)


if __name__=='__main__':
	args = parse_args()
	run(args)