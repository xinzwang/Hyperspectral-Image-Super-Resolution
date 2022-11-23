import torch
import math
import torch.nn as nn
from .common import *

class CANetV1(nn.Module):
	def __init__(self, channels, scale_factor):
		super(CANetV1, self).__init__()
		self.channels = channels
		self.scale_factor = scale_factor
		
		n_feats = 256
		n_subs = 8	# channels for group

		n_blocks = 3

		conv = default_conv

		# calculate group index
		# self.group_idx = []
		# self.G = math.ceil(channels / n_subs)
		# for g in range(self.G):
		# 	s_idx = g * n_subs
		# 	e_idx = min(s_idx + n_subs, channels)
		# 	self.group_idx.append((s_idx, e_idx))


		self.head = nn.Sequential(
			conv(channels, n_feats, kernel_size=3, bias=True),
		)

		self.body = SSPN(n_feats, n_blocks, act=nn.ReLU(True), res_scale=0.1)

		self.up = Upsampler(conv, scale_factor, n_feats)

		self.trunk = SSPN(n_feats, n_blocks, act=nn.ReLU(True), res_scale=0.1)

		self.tail = nn.Sequential(
			conv(n_feats, channels, kernel_size=3, bias=True),
		)
		

	def forward(self, x):
		N, C, H, W = x.shape

		x1 = self.head(x)

		# x2 = torch.zeros(N, C, H * self.scale_factor, W*self.scale_factor).to(x.device)
		# for s_idx, e_idx in self.group_idx:
		# 	xi = x1[:, s_idx:e_idx, :, :]
		# 	xi = self.branch(xi)
		# 	x2[:, s_idx:e_idx, :, :] += xi
		x2 = self.body(x1)

		x3 = self.up(x2)
		x4 = self.trunk(x3) + x3
		x5 = self.tail(x4)
		return x5