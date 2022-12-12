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
		self.layer2 = CALayer2DFeatsDimWithChansPosEmbed(channels=channels, n_feats=n_feats, reduction=reduction)

	def forward(self, x, mode):
		x = self.layer2(self.layer1(x), rgb_mode=(mode=='RGB')) + x
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

		self.sa_hsi = BandSALayer3D(channels, n_feats, cfgs, act=act)
		self.sa_rgb = BandSALayer3D(3, n_feats, cfgs, act=act)

	def forward(self, x, mode='HSI'):
		out = x
		for layer in self.layers:
			out = layer(out, mode=mode)
		out = self.sa_hsi(out) if mode=='HSI' else self.sa_rgb(out)
		return out + x


class OursV3(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		n_feats = 64
		embed_chans = 3
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
		self.hsi_mean = torch.autograd.Variable(torch.FloatTensor(band_means['CAVE'])).view([1, channels, 1, 1])
		self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(band_means['DIV2K'])).view([1, 3, 1, 1])
		# head
		self.hsi_head = nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2))
		self.rgb_head = nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2))
		# body
		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(cfgs))
		self.conv_after_body = nn.Conv3d(n_feats, n_feats, kernel_size=(1,1,1))

		# tail
		self.hsi_tail = nn.Sequential(
			# nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1)),
			# nn.ReLU(inplace=True),
			nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(1,1,1)),
			nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)
		)
		self.rgb_tail = nn.Sequential(
			# nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1)),
			# nn.ReLU(inplace=True),
			nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(1,1,1)),
			nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)
		)

	def forward(self, x):
		return self.forward_hsi(x)

	def forward_hsi(self, x):
		out = x - self.hsi_mean.to(x.device)
		out = out.unsqueeze(1)
		# head
		out_head = self.hsi_head(out)
		# body
		out = out_head
		for block in self.blocks:
			out = block.forward(out, mode='HSI')
		out = self.conv_after_body(out) + out_head
		# tail
		out = self.hsi_tail(out)

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
			out = block.forward(out, mode='RGB')
		out = self.conv_after_body(out) + out_head
		# tail
		out = self.hsi_tail(out)

		out = out.squeeze(1)
		out = out + self.rgb_mean.to(x.device)
		return out


