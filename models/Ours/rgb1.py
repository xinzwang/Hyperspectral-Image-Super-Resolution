import torch
import torch.nn as nn

from .common import *


class Layer(nn.Module):
	def __init__(self, cfgs):
		super().__init__()
		self.cfgs = cfgs

		channels = cfgs['channels']
		n_feats = cfgs['n_feats']
		reduction = cfgs['reduction']
		act = cfgs['act']

		self.layer1 = DoubleConv2DFeatsDim(channels, n_feats, act=act)
		# self.layer2 = CALayer(channel=n_feats, reduction=reduction)
	
	def forward(self, x):
		out = self.layer1(x) + x
		return out


class Block(nn.Module):
	def __init__(self, cfgs):
		super().__init__()
		self.channels = cfgs['channels']
		self.n_feats = cfgs['n_feats']

		n_feats = cfgs['n_feats']
		n_layers = cfgs['n_layers']

		layers = []
		for i in range(n_layers):
			layers.append(Layer(cfgs))

		self.layers = nn.Sequential(*layers)

		self.hsi_conv = nn.Conv3d(n_feats, n_feats, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
		self.rgb_conv = nn.Conv3d(n_feats, n_feats, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
		
	def forward_hsi(self, x):
		out = self.layers(x)
		out = self.hsi_conv(out) + x
		return out

	def forward_rgb(self, x):
		out = self.layers(x)
		out = self.rgb_conv(out) + x
		return out


class OursRGB1(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		n_feats = 64
		embed_chans = 3
		kernel_size=3
		
		n_blocks = 4
		n_layers = 3

		cfgs = {}
		cfgs['channels'] = channels
		cfgs['scale_factor'] = scale_factor
		cfgs['kernel_size'] = kernel_size
		cfgs['n_feats'] = n_feats
		cfgs['n_layers'] = n_layers
		cfgs['reduction'] = 16
		cfgs['act'] = nn.LeakyReLU(inplace=True)

		# mean
		self.hsi_mean = torch.autograd.Variable(torch.FloatTensor(band_means['CAVE'])).view([1, channels, 1, 1])
		self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(band_means['DIV2K'])).view([1, 3, 1, 1])
		# head
		self.hsi_head = nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2))
		self.rgb_head = nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2))
		# body
		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(cfgs))
		# tail
		self.hsi_tail = nn.Sequential(
			nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(1,1,1)),
			nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)
		)
		self.rgb_tail = nn.Sequential(
			nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(1,1,1)),
			nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)
		)


	def forward_hsi(self, x):
		out = x - self.hsi_mean.to(x.device)
		out = out.unsqueeze(1)
		# head
		out_head = self.hsi_head(out)
		# body
		out = out_head
		for block in self.blocks:
			out = block.forward_hsi(out)
		# tail
		out = self.hsi_tail(out + out_head)

		out = out.squeeze(1)
		out = out + self.hsi_mean.to(x.device)
		return out

	def forward_rgb(self, x):
		out = x - self.rgb_mean.to(x.device)
		out = out.unsqueeze(1)
		# head
		out_head = self.rgb_head(out)
		# body
		out = out_head
		for block in self.blocks:
			out = block.forward_rgb(out)
		# tail
		out = self.rgb_tail(out + out_head)

		out = out.squeeze(1)
		out = out + self.rgb_mean.to(x.device)
		return out


