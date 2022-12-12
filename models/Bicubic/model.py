import torch
import torch.nn as nn

class Bicubic(nn.Module):
	def __init__(self, channels, scale_factor, align_corners=False):
		super().__init__()

	def forward(self, x, lms):
		return lms