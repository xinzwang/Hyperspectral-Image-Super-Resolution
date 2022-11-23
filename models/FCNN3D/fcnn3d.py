import torch
import torch.nn as nn


class FCNN3D(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		self.up = nn.Upsample(
			scale_factor=scale_factor,
			mode='bicubic',
			align_corners=True
		)
		self.conv = nn.Sequential(
			nn.Conv3d(1, 64, kernel_size=(7, 9, 9), padding=(7//2, 9//2, 9//2)),
			nn.ReLU(inplace=True),
			nn.Conv3d(64,32,kernel_size=(1, 1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, 9, kernel_size=(1, 1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(9, 1, kernel_size=(3, 5, 5), padding=(3//2, 5//2, 5//2))
		)

	def forward(self, x):
		out = self.up(x)
		out = out.unsqueeze(dim=1) # (N,C,H,W)->(N,1,C,H,W)
		out = self.conv(out)
		out = out.squeeze(dim=1)
		return out
