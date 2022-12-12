"""
Test api
"""
import os
import cv2
import time
import numpy as np
from tqdm import tqdm
import torch

from .eval_metrics import cal_metrics, cal_metrics2

def test(model, dataloader, device, rgb_mode=None, use_lms=False):
	psnr, ssim, sam, ergas = [], [], [], []
	run_time = 0
	# forward all datas
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(tqdm(dataloader)):
			if use_lms:
				lr, lms, hr = data
				lms = lms.to(device)
			else:
				lr, hr = data
			lr = lr.to(device)

			start_time = time.time()
			if rgb_mode is None:
				pred = model(lr, lms) if use_lms else model(lr)
			else:
				pred = model.forward_hsi(lr)
			end_time = time.time()
			run_time += end_time-start_time

			# torch->numpy; 1CHW->HWC; [0, 1]
			psnr_, ssim_, sam_, ergas_ = cal_metrics(
				pred=pred.cpu().numpy().transpose(0, 2, 3, 1),
				hr=hr.cpu().numpy().transpose(0, 2, 3, 1),
				max_v=1.0)
			psnr += psnr_
			ssim += ssim_
			sam += sam_
			ergas += ergas_
	
	# cal metrics
	psnr, ssim, sam, ergas = np.mean(psnr), np.mean(ssim), np.mean(sam), np.mean(ergas)
	print('[TEST] runtime:%.2fs psnr:%.7f ssim:%.7f sam:%.7f ergas:%.7f' %(run_time, psnr, ssim, sam, ergas))
	return psnr, ssim, sam, ergas


def visual(model, dataloader, img_num=3, save_path='img/', err_gain=10, device=None, rgb_mode=None, use_lms=False):
	# create save dir
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# infer and save
	it = iter(dataloader)
	for i in range(min(img_num, dataloader.__len__())):
		if use_lms:
			lr, lms, hr = next(it)
			lms = lms.to(device)
		else:
			lr, hr = next(it)
		lr = lr.to(device)

		with torch.no_grad():
			if rgb_mode is None:
				pred = model(lr, lms) if use_lms else model(lr)
			else:
				pred = model.forward_hsi(lr)
		# assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
		# torch->numpy; 1CHW->HWC; [0, 1]
		hr_ = hr.cpu().numpy()[0].transpose(1,2,0)
		pred_ = pred.cpu().numpy()[0].transpose(1,2,0)
		# save err_map gray
		err = np.mean(np.abs(pred_ - hr_), axis=2)
		gray = np.mean(pred_, axis=2)
		cv2.imwrite(save_path + '%d_err.png' %(i), cv2.applyColorMap((err * 255 * err_gain).astype(np.uint8), cv2.COLORMAP_JET))
		cv2.imwrite(save_path + '%d_gray.png'%(i), gray * 255.0)
	return










