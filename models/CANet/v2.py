import torch
import math
import torch.nn as nn
from .common import *

class FusCALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(FusCALayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv_du = nn.Sequential(
      nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
      nn.Sigmoid()
    )
	
	def forward(self, x):
		# x[0]: body branch; x[1]: feat branch
		body, feat = x
		a = self.avg_pool(feat)
		a = self.conv_du(a)
		return body * a


class FeatBlock(nn.Module):
	def __init__(self, n_feats):
		super(FeatBlock, self).__init__()

		conv = default_conv
		act=nn.ReLU(inplace=True)
		res_scale = 0.1

		self.spa = ResBlock(conv, n_feats, kernel_size=3, act=act, res_scale=res_scale)
		# self.spc = ResAttentionBlock(conv, n_feats, kernel_size=1, act=act, res_scale=res_scale)

	def forward(self, x):
		# return self.spc(self.spa(x))
		return self.spa(x)


class BodyBlock(nn.Module):
	def __init__(self, n_feats):
		super(BodyBlock, self).__init__()

		conv = default_conv
		act=nn.ReLU(inplace=True)
		res_scale = 0.1
		reduction = 16

		self.fus = FusCALayer(n_feats, reduction=reduction)

		# self.spa = ResBlock(conv, n_feats, kernel_size=3, act=act, res_scale=res_scale)
		self.spc = ResAttentionBlock(conv, n_feats, kernel_size=3, act=act, res_scale=res_scale)
		
		# self.conv = default_conv(n_feats*2, n_feats, kernel_size=1, bias=False)

	def forward(self, x):
		x_b, x_f = x

		out = self.fus((x_b, x_f)) + x_b
		# out = torch.cat((x_b, x_f), 1)
		# out = self.conv(out) + x_b	# TODO: out + x_b + x_f
		# out = self.conv(out)

		# out = self.spa(out)
		out = self.spc(out)
		return out	

	

class CANetV2(nn.Module):
	def __init__(self, channels, scale_factor):
		super(CANetV2, self).__init__()

		n_feats = 256

		n_blocks = 6

		self.n_blocks = n_blocks

		conv = default_conv

		self.head = nn.Sequential(
			conv(channels, n_feats, kernel_size=3, bias=False)
		)

		self.feat_blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.feat_blocks.append(FeatBlock(n_feats))

		self.body_blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.body_blocks.append(BodyBlock(n_feats))

		self.up = Upsampler(default_conv, scale_factor, n_feats)

		self.tail = nn.Sequential(
			conv(n_feats, channels, kernel_size=3, bias=False),
		)
		return
	
	def forward(self, x):

		out = self.head(x)
		
		out_f = out
		out_b = out
		for i in range(self.n_blocks):
			feat_block = self.feat_blocks[i]
			body_block = self.body_blocks[i]

			out_f = feat_block(out_f)
			out_b = body_block((out_b, out_f))

		out = self.up(out_b + out)
		out = self.tail(out)
		return out



