import cv2
import numpy as np


# Gen train dataset
def gen_panel_from_area(area, panel_size=64, stride=64, ratio=1, rot=None):
	# down sample
	if ratio != 1:
		assert ratio < 1, Exception('Ratio should be less than 1. Upsampling is not recommended')
		area = cv2.resize(area, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
	
	# Rotate image for k*pi/2. k=[0,1,2,3]
	if rot is not None:
		area = np.rot90(area, rot)

	H, W, C = area.shape

	if not(panel_size<H and panel_size<W):
		return []
		# Exception('Panelsize is too large')

	P, Q = H // panel_size, W // panel_size
	res = []
	for i in range(P):
		for j in range(Q):
			x, y = i*panel_size, j * panel_size
			t = area[x:x+panel_size, y:y+panel_size, ...]
			res.append(t)
	return res


# Norm
def norm(img):
	max_ = img.max()
	min_ = img.min()
	return (img - min_) / (max_-min_)


# down sample
def down_sample(x, scale_factor=2, kernel_size=(7,7), sigma=3):
	out = cv2.GaussianBlur(x, ksize=kernel_size, sigmaX=sigma,sigmaY=sigma)
	out = cv2.resize(out, (0,0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_CUBIC)
	return out

def bicubic_downsample(img, scale_factor=2):
  [h, w, _] = img.shape
  new_h, new_w = int(h / scale_factor), int(w / scale_factor)
  return cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)