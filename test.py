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

import utils.test_rgb as test_rgb
from utils.tools import build_dataset
from utils.test import test, visual
from utils.core_rgb import RGBCore

from models import *
# from models import BiFQRNNREDC3D

def parse_args():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--mode', type=str, choices=['RGB', 'HSI'], required=True)
	parser.add_argument('--scale_factor', default=4, type=int)
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--device', default='cuda:1')
	parser.add_argument('--visual', default=False)
	parser.add_argument('--save_path', default='imgs/')

	# parser.add_argument('--ckpt', default='Bicubic')
	parser.add_argument('--ckpt', type=str)
	parser.add_argument('--use_lms', type=bool)

	parser.add_argument('--dataset', default='CAVE')

	args = parser.parse_args()
	print(args)
	return args


def run(args):
	device=torch.device(args.device)

	test_dataset, test_dataloader = build_dataset(
		name=args.dataset,
		scale_factor=args.scale_factor, 
		state='test',
		use_lms=args.use_lms)

	# load ckpts
	if args.ckpt == 'Bicubic':
		model = Bicubic(31, args.scale_factor)
	else:
		ckpt = torch.load(args.ckpt, map_location=device)
		model = ckpt['model'].to(device)

	# forward test
	if args.dataset in ('CAVE', 'Harvard', 'ICVL'):
		test(model, test_dataloader, device, use_lms=args.use_lms)
	elif args.dataset in ('Set5', 'Set14'):
		psnr, ssim, _ = test_rgb.test(model, test_dataloader, device)

	# visual
	if args.visual:
		visual(model, test_dataloader, img_num=3, save_path=save_path, device=device)


if __name__=='__main__':
	args = parse_args()
	run(args)