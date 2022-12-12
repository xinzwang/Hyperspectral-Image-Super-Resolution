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
from utils.core_rgb import RGBCore

import utils.test_rgb as test_rgb

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['RGB', 'HSI', 'Joint'])	
	parser.add_argument('--scale_factor', default=4, type=int)
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--epoch', default=2000, type=int)
	parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
	parser.add_argument('--seed', default=17, type=int)
	parser.add_argument('--device', default='cuda:0')
	parser.add_argument('--parallel', default=False, type=bool)
	parser.add_argument('--device_ids', default=['cuda:2', 'cuda:4', 'cuda:0', 'cuda:3'])
	parser.add_argument('--loss', default='L1', choices=['L1', 'L2', 'Hybrid'])
	parser.add_argument('--model', default='OursRGB1')

	parser.add_argument('--use_ckpt', default=False, type=bool)
	parser.add_argument('--ckpt_path', default='ckpts/CAVE/OursRGB21/4/2022-11-29_13-23-40/ckpts/epoch=1200/ckpt.pt')

	# CAVE HSI dataset
	parser.add_argument('--dataset', default='CAVE')
	parser.add_argument('--train_path', default=('data/CAVE/archive/bicubic_gauss_ksize=9_sigma=3/train_scale=4.npy', 'data/CAVE/archive/train.npy'))
	parser.add_argument('--val_path',   default=('data/CAVE/archive/bicubic_gauss_ksize=9_sigma=3/val_scale=4.npy', 'data/CAVE/archive/val.npy'))

	# ICVL HSI dataset
	# parser.add_argument('--train_path', default='')
	# parser.add_argument('--val_path',   default=('data/CAVE/archive/val.npy', 'data/CAVE/archive/bicubic_gauss_ksize=9_sigma=3/val_scale=4.npy'))


	# DIV2K RGB dataset
	parser.add_argument('--rgb_dataset', default='DIV2K')
	parser.add_argument('--rgb_train_path', default=('../super_resolution/data/DIV2K/DIV2K_train_LR_bicubic/X2/',))
	parser.add_argument('--rgb_val_path',   default=('../super_resolution/data/DIV2K/DIV2K_train_LR_bicubic/X2/',))
	# parser.add_argument('--rgb_train_path', default=('../super_resolution/data/DIV2K/DIV2K_train_HR/', '../super_resolution/data/DIV2K/DIV2K_train_LR_bicubic/X4/'))
	# parser.add_argument('--rgb_val_path',   default=('../super_resolution/data/DIV2K/DIV2K_train_HR/', '../super_resolution/data/DIV2K/DIV2K_train_LR_bicubic/X4/'))

	# parser.add_argument('--rgb_dataset', default='CAVE_rgb')
	# parser.add_argument('--rgb_train_path', default=('data/CAVE/archive/rgb/train/',))
	# parser.add_argument('--rgb_val_path',   default=('data/CAVE/archive/rgb/val/',))

	args = parser.parse_args()
	print(args)
	return args

def train(args):
	t = time.strftime('%Y-%m-%d_%H-%M-%S')
	checkpoint_path = 'ckpts/%s/%s/%d/%s/%s/' % (args.dataset, args.model, args.scale_factor, args.mode, t)
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

	# core
	hsi_chans = 31 if args.dataset in ['CAVE', 'ICVL'] else None
	core = RGBCore(batch_log=10)
	core.inject_logger(logger)
	core.inject_writer(writer)
	core.inject_device(device)
	if args.use_ckpt == True:
		core.load_ckpt(path=args.ckpt_path)
	else:
		core.build_model(name=args.model, channels=hsi_chans, scale_factor=args.scale_factor)
	
	# RGB dataset
	if args.mode == 'RGB' or args.mode == 'Joint':
		rgb_dataset, rgb_dataloader = build_dataset(dataset=args.rgb_dataset, path=args.rgb_train_path,batch_size=args.batch_size, scale_factor=args.scale_factor, test_flag=False)
	rgb_test_dataset, rgb_test_dataloader = build_dataset(dataset=args.rgb_dataset, path=args.rgb_val_path,batch_size=1, scale_factor=args.scale_factor, test_flag=True)

	# HSI dataset
	if args.mode == 'HSI' or args.mode == 'Joint':
		hsi_dataset, hsi_dataloader = build_dataset(dataset=args.dataset, path=args.train_path, batch_size=args.batch_size, scale_factor=args.scale_factor, test_flag=False)
	hsi_test_dataset, hsi_test_dataloader = build_dataset(dataset=args.dataset, path=args.val_path,batch_size=1, scale_factor=args.scale_factor, test_flag=True)
	
	if args.mode == 'Joint':
		train_dataloader = (hsi_dataloader, rgb_dataloader)
	elif args.mode == 'HSI':
		train_dataloader = hsi_dataloader
	elif args.mode == 'RGB':
		train_dataloader = rgb_dataloader


	# loss optimizer
	if args.loss == 'L1':
		loss_fn = nn.L1Loss()
	elif args.loss == 'L2':
		loss_fn = nn.MSELoss()
	elif args.loss == 'Hybrid':
		loss_fn = HybridLoss(spatial_tv=True, spectral_tv=True)	# This loss is refs to SSPSR
	else:
		raise Exception('Unknown loss type:'+args.loss)
		
	optimizer = optim.Adam(core.model.parameters(), args.lr)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, min_lr=1e-5)
	# scheduler = optim.lr_scheduler.StepLR(optimizer, 200 * 20, gamma=0.1)
	core.inject_loss_fn(loss_fn)
	core.inject_optim(optimizer)
	# core.inject_scheduler(scheduler)

	if args.parallel:
		core.parallel(device_ids = args.device_ids)
	
	# train loop
	for epoch in range(args.epoch):
		mean_loss = core.train(train_dataloader, mode=args.mode)
		logger.info('[TEST] epoch:%d mean_loss:%.5f'%(epoch, mean_loss))

		if epoch % (10 if args.mode=='RGB' else 10) == 0:
			save_path = checkpoint_path + 'ckpts/epoch=%d/' % (epoch)
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			# HSI
			# if args.mode == 'HSI' or args.mode == 'Joint':
			psnr, ssim, sam, ergas = test(core.model, hsi_test_dataloader, device=device, rgb_mode=(args.mode=='RGB'))
			logger.info('[TEST] epoch:%d psnr:%.7f ssim:%.7f sam:%.7f mergas:%.7f' %(epoch, psnr, ssim, sam, ergas))
			writer.add_scalar(tag='score/PSNR', scalar_value=psnr, global_step=epoch)
			writer.add_scalar(tag='score/SSIM', scalar_value=ssim, global_step=epoch)
			writer.add_scalar(tag='score/SAM', scalar_value=sam, global_step=epoch) 
			writer.add_scalar(tag='score/ERGAS', scalar_value=ergas, global_step=epoch)
			visual(core.model, hsi_test_dataloader, img_num=3, save_path=save_path+'hsi_psnr=%.2f_ssim=%.2f_sam=%.2f/'%(psnr, ssim, sam), device=device, rgb_mode=(args.mode=='RGB'))
			# RGB
			psnr, ssim, _ = test_rgb.test(core.model, rgb_test_dataloader, device)
			test_rgb.infer_img(save_path+'rgb_psnr=%.2f_ssim=%.2f/' %(psnr, ssim), num=9, model=core.model, dataloader=rgb_test_dataloader,device=device)
			writer.add_scalar(tag='score/rgb_PSNR', scalar_value=psnr, global_step=epoch)
			writer.add_scalar(tag='score/rgb_SSIM', scalar_value=ssim, global_step=epoch)
			# Save checkpoint
			core.save_ckpt(save_path +'ckpt.pt', dataset=args.dataset)
	return


if __name__=='__main__':
	args = parse_args()
	train(args)