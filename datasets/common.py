"""
Common functions for data process
"""

###################### Downsample ######################
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom


# def Downsample(img, scale_factor, sigma=3, ksize=8):
# 	"""Input: (H, W, C)"""
# 	img = gaussian_filter(img, sigma=(sigma, sigma, 1.), truncate=((ksize/2-1)/sigma))
# 	img = zoom(img, (1./scale_factor, 1./scale_factor, 1))
# 	return img

def Downsample(img, scale_factor, sigma=3, ksize=9):
	"""Input: (H, W, C)"""
	out = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma,sigmaY=sigma)
	out = cv2.resize(out, (0,0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_CUBIC)
	return out


###################### Enhance ######################
import random

def RandomPatch(img_in, img_tar, patch_size, scale, multi_scale=False):
		ih, iw = img_in.shape[:2]

		p = scale if multi_scale else 1
		tp = p * patch_size
		ip = tp // scale

		ix = random.randrange(0, iw - ip + 1)
		iy = random.randrange(0, ih - ip + 1)
		tx, ty = scale * ix, scale * iy

		img_in = img_in[iy:iy + ip, ix:ix + ip, :]
		img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

		return img_in, img_tar

def CenterCrop(img, size=512):
	H, W, C = img.shape
	center_H = H // 2
	center_W = W // 2
	R = size // 2
	return img[center_H-R:center_H+R, center_W-R:center_W+R, :]

def Augment(imgs, hflip=True, rot=True):	
	if not isinstance(imgs, list):
		imgs = [imgs]

	hflip = hflip and random.random() < 0.5
	vflip = rot and random.random() < 0.5
	rot90 = rot and random.random() < 0.5

	def _augment(img):
		if img is None: return None
		if hflip: img = img[:, ::-1, :]
		if vflip: img = img[::-1, :, :]
		if rot90: img = img.transpose(1, 0, 2)
			
		return img
	return [_augment(img) for img in imgs]


if __name__=='__main__':
	import cv2
	import numpy as np
	from glob import glob

	paths = glob('/data2/wangxinzhe/codes/datasets/Set14/image_SRF_4/hr/*.png')
	for i, p in enumerate(paths):
		hr = cv2.imread(p)
		lr = Downsample(hr, 4, sigma=3, ksize=9)
		# cv2.imwrite('img/hr_%d.png'%(i), hr)
		cv2.imwrite('img/lr_noblur_%d.png'%(i), lr)

