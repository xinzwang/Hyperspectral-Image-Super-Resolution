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

from models.loss import HybridLoss

from utils.tools import create_logger, build_dataset, set_seed
from utils.test import test, visual
from utils.core import SRCore

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CAVE', choices=['CAVE', 'Pavia', 'Salinas','PaviaU', 'KSC', 'Indian', 'ICVL'])
	parser.add_argument('--scale_factor', default=4, type=int)
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--epoch', default=2000)
	parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
	parser.add_argument('--seed', default=17, type=int)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--parallel', default=False, type=bool)
	parser.add_argument('--device_ids', default=['cuda:2', 'cuda:4', 'cuda:0', 'cuda:3'])
	parser.add_argument('--loss', default='L1')
	parser.add_argument('--model', default='SSPSR')
	# CAVE
	parser.add_argument('--train_hr_path', default='data/CAVE/train.npy')
	parser.add_argument('--train_lr_path', default='data/CAVE/train_scale=4_ksize=9_sigma=3.npy')
	parser.add_argument('--val_hr_path', default='data/CAVE/val.npy')
	parser.add_argument('--val_lr_path', default='data/CAVE/val_scale=4_ksize=9_sigma=3.npy')

	args = parser.parse_args()
	print(args)
	return args

def train(args):
	t = time.strftime('%Y-%m-%d_%H-%M-%S')
	checkpoint_path = 'ckpts/%s/%s/%d/%s/' % (args.dataset, args.model, args.scale_factor, t)
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
		
	logger = create_logger(checkpoint_path + 'logs.log')
	logger.info(str(args))

	writer = SummaryWriter(checkpoint_path)

	# set seed
	set_seed(args.seed)

	# device
	cudnn.benchmark = True
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	# dataset
	dataset, dataloader = build_dataset(
		dataset=args.dataset, 
		path=(args.train_lr_path, args.train_hr_path),
		batch_size=args.batch_size, 
		scale_factor=args.scale_factor, 
		test_flag=False)
	test_dataset, test_dataloader = build_dataset(
		dataset=args.dataset, 
		path=(args.val_lr_path, args.val_hr_path),
		batch_size=1, 
		scale_factor=args.scale_factor, 
		test_flag=True)

	# core
	core = SRCore(batch_log=10)
	core.inject_logger(logger)
	core.inject_writer(writer)
	core.inject_device(device)
	core.build_model(name=args.model, channels=dataset.channels, scale_factor=args.scale_factor)
	
	# loss optimizer
	if args.loss == 'L1':
		loss_fn = nn.L1Loss()
	elif args.loss == 'Hybrid':
		loss_fn = HybridLoss(spatial_tv=True, spectral_tv=True)	# This loss is refs to SSPSR
	else:
		raise Exception('Unknown loss type:'+args.loss)
		
	optimizer = optim.Adam(core.model.parameters(), args.lr)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, threshold=1e-4, min_lr=1e-5)
	scheduler = optim.lr_scheduler.StepLR(optimizer, 35, gamma=0.5)
	core.inject_loss_fn(loss_fn)
	core.inject_optim(optimizer)
	core.inject_scheduler(scheduler)

	if args.parallel:
		core.parallel(device_ids = args.device_ids)
	
	# train loop
	for epoch in range(args.epoch):
		mean_loss = core.train(dataloader)
		logger.info('[TEST] epoch:%d mean_loss:%.5f'%(epoch, mean_loss))

		if epoch % 5 == 0:
			psnr, ssim, sam, ergas = test(core.model, test_dataloader, device=device)
			logger.info('[TEST] epoch:%d psnr:%.7f ssim:%.7f sam:%.7f mergas:%.7f' %(epoch, psnr, ssim, sam, ergas))
			writer.add_scalar(tag='score/PSNR', scalar_value=psnr, global_step=epoch)
			writer.add_scalar(tag='score/SSIM', scalar_value=ssim, global_step=epoch)
			writer.add_scalar(tag='score/SAM', scalar_value=sam, global_step=epoch)
			writer.add_scalar(tag='score/ERGAS', scalar_value=ergas, global_step=epoch)

			save_path = checkpoint_path + 'epoch=%d_psnr=%.2f_ssim=%.2f/'%(epoch, psnr,ssim)
			visual(core.model, test_dataloader, img_num=3, save_path=save_path, device=device)
			core.save_ckpt(save_path +'ckpt.pt', dataset=args.dataset)

	# save the final model
	save_path = checkpoint_path + 'final_psnr=%.2f_ssim=%.2f/'%(epoch, psnr, ssim)
	visual(core.model, test_dataloader, img_num=3, save_path=save_path+'img/')
	core.save_ckpt(save_path +'.pt', dataset=args.dataset)
	return


if __name__=='__main__':
	args = parse_args()
	train(args)