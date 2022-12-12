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

		self.layer1 = DoubleConv2DFeatsDim(channels, n_feats, kernel_size=3, act=act)
		self.layer2 = CALayer2DFeatsDim(channels=channels, n_feats=n_feats, reduction=reduction)
		# self.layer2 = DoubleConv2DFeatsDim(channels, n_feats, kernel_size=1, act=act)
		# self.layer3 = CALayer2DFeatsDimWithChansPosEmbed(channels=channels, n_feats=n_feats, reduction=reduction, act=act)
	
	def forward(self, x, mode=None):
		x = self.layer2(self.layer1(x)) + x
		return x


class Block(nn.Module):
	def __init__(self, cfgs):
		super().__init__()
		self.channels = cfgs['channels']
		self.n_feats = cfgs['n_feats']

		channels = cfgs['channels']
		n_feats = cfgs['n_feats']
		n_layers = cfgs['n_layers']
		act = cfgs['act']
		
		self.layers = nn.ModuleList()
		for i in range(n_layers):
			self.layers.append(Layer(cfgs))
	
	def forward(self, x, mode):
		out = x
		for layer in self.layers:
			out = layer(out, mode=mode)
		return out + x


class PerBand(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		n_feats = 64
		kernel_size=3
		
		n_blocks = 6
		n_layers = 3

		cfgs = {}
		cfgs['channels'] = channels
		cfgs['scale_factor'] = scale_factor
		cfgs['kernel_size'] = kernel_size
		cfgs['n_feats'] = n_feats
		cfgs['n_layers'] = n_layers
		cfgs['reduction'] = 16
		cfgs['act'] = nn.ReLU(inplace=True)

		cfgs['attn_drop'] = 0.0
		cfgs['proj_drop'] = 0.0
		cfgs['n_heads'] = 4

		# mean
		# self.hsi_mean = torch.autograd.Variable(torch.FloatTensor(band_means['CAVE'])).view([1, channels, 1, 1])
		# self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(band_means['DIV2K'])).view([1, 3, 1, 1])
		# self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([0.5])).view([1, 1, 1, 1])
		# head
		self.head = nn.Conv3d(1, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1))

		# body
		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(cfgs))
		self.conv_after_body = nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1))

		# tail
		self.tail = nn.Sequential(
			nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(1,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(0,1,1)),
			nn.Conv3d(n_feats, 1, kernel_size=(1,3,3), padding=(0, 1, 1))
		)

	def forward(self, x, mode='HSI'):
		# x = x - (self.hsi_mean if mode == 'HSI' else self.rgb_mean).to(x.device)
		x = x.unsqueeze(1)

		# Shallow Extraction
		head = self.head(x)
		
		# Deep Feature Extraction
		x = head
		for block in self.blocks:
			x = block(x, mode=mode)
		x = self.conv_after_body(x) + head
		# Upsample and post process
		x = self.tail(x)

		x = x.squeeze(1)
		# x = x + (self.hsi_mean if mode == 'HSI' else self.rgb_mean).to(x.device)
		return x
	
	def forward_hsi(self, x):
		return self.forward(x, mode='HSI')

	def forward_rgb(self, x):
		return self.forward(x, mode='RGB')

