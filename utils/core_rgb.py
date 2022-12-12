"""
Core for training assembly
"""
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle

import models


class RGBCore:
	def __init__(self, batch_log=10):
		self.parallel_ = False
		self.batch_log = batch_log
		self.batch_cnt = 0
		self.epoch_cnt = 0
		self.scheduler = None
		pass

	def inject_logger(self, logger):
		self.logger = logger

	def inject_writer(self, writer):
		self.writer = writer

	def inject_device(self, device):
		self.device=device

	def build_model(self, name, channels=3, scale_factor=2):
		self.model = getattr(models, name)(channels, scale_factor).to(self.device)
		total = sum([param.nelement() for param in self.model.parameters()])
		print("Number of parameter: %.2fM" % (total/1e6))
	
	def load_ckpt(self, path):
		ckpt = torch.load(path, map_location=self.device)
		print('[Core] load ckpt. Dataset:', ckpt['model'])
		self.model = ckpt['model'].to(self.device)
		return
	
	def build_loss(self):
		self.loss_fn = getattr(self.model.loss_name, )

	def inject_loss_fn(self, loss_fn):
		self.loss_fn = loss_fn

	def inject_optim(self, optimizer):
		self.optimizer = optimizer

	def inject_scheduler(self, scheduler):
		self.scheduler = scheduler
	
	def parallel(self, device_ids=['cuda:0']):
		if self.parallel_==False:
			self.parallel_ = True
			self.model = nn.DataParallel(self.model, device_ids=device_ids)
			self.optimizer = nn.DataParallel(self.optimizer, device_ids=device_ids)
			self.scheduler = nn.DataParallel(self.scheduler, device_ids=device_ids)

	def train(self, dataloader, mode='HSI'):
		if self.parallel_==True:
			# mean_loss = self.train_parallel(dataloader)
			pass
		else:
			mean_loss = self.train_single(dataloader, mode=mode)
		return mean_loss

	def step_rgb(self, lr, hr):
		self.optimizer.zero_grad()
		pred = self.model.forward_rgb(lr)
		loss = self.loss_fn(pred, hr)
		loss.backward()
		self.optimizer.step()
		return loss

	def step_hsi(self, lr, hr):
		self.optimizer.zero_grad()
		pred = self.model.forward_hsi(lr)
		loss = self.loss_fn(pred, hr)
		loss.backward()
		self.optimizer.step()
		return loss

	def rgb_img(self, rgbIter):
		try:
			rgb_lr, rgb_hr = next(rgbIter)
		except StopIteration:
			rgbIter = iter(rgb_dataloader)
			rgb_lr, rgb_hr = next(rgbIter)
		rgb_lr = rgb_lr.to(self.device)
		rgb_hr = rgb_hr.to(self.device)
		return rgb_lr, rgb_hr

	def train_single(self, dataloader, mode='HSI'):
		logger = self.logger
		writer = self.writer
		device = self.device
		model = self.model
		loss_fn = self.loss_fn
		optimizer = self.optimizer
		scheduler = self.scheduler

		self.epoch_cnt += 1

		total_loss = []
		c_lr = optimizer.state_dict()['param_groups'][0]['lr']
		logger.info('  lr:%f'%(c_lr))
		writer.add_scalar(tag='train/lr', scalar_value=c_lr, global_step=self.epoch_cnt)

		if mode == 'Joint':
			dataloader, rgb_dataloader = dataloader
			rgbIter = cycle(iter(rgb_dataloader))

		for i, (lr, hr) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)
			hr = hr.to(device)

			if mode == 'RGB':
				loss = self.step_rgb(lr, hr)
			elif mode == 'HSI':
				loss = self.step_hsi(lr, hr)
			elif mode == 'Joint':
				# 1. HSI Step
				loss = self.step_hsi(lr, hr)
				# 2. RGB Step
				rgb_lr, rgb_hr = self.rgb_img(rgbIter)
				self.step_rgb(rgb_lr, rgb_hr)
			else:
				raise Exception('Unknown mode:', mode)

			total_loss.append(loss.item())
			self.batch_cnt += 1
			if i % self.batch_log == 1:
				if mode == 'RGB':
					logger.info('  RGB batch:%d loss:%.5f' % (i, loss.item()))
					writer.add_scalar(tag='train/rgb_loss', scalar_value=loss.item(), global_step=self.batch_cnt)
				else:
					logger.info('  HSI batch:%d loss:%.5f' % (i, loss.item()))
					writer.add_scalar(tag='train/loss', scalar_value=loss.item(), global_step=self.batch_cnt)
			pass
		mean_loss = np.mean(total_loss)
		if scheduler is not None:
			if isinstance(scheduler, optim.lr_scheduler.StepLR):
				scheduler.step()
			else:
				scheduler.step(mean_loss)
			
		return mean_loss

	def save_ckpt(self, save_path, dataset):
		torch.save({
			'dataset': dataset,
			'model': self.model
		}, save_path)
		return

	def predict(self, dataloader):
		device = self.device
		model = self.model

		err_channel = None

		for i, (lr, hr) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)
			hr = hr.to(device)

			with torch.no_grad():
				pred = model(lr)
			# cal error
			pred_ = pred.cpu().numpy()
			hr_ = hr.cpu().numpy()
			err_ = np.abs(pred_ - hr_)
			err_c = np.mean(err_, axis=(2, 3))
			if err_channel is None:
				err_channel = err_c
			else:
				err_channel = np.concatenate((err_channel, err_c), axis=0)
		return err_channel

