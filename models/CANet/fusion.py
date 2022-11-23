import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, default_conv

class FusAffineLayer(nn.Module):
	def __init__(self, n_feats, knn_feats, hidden_feats=64, conv=default_conv):
		super(FusAffineLayer, self).__init__()

		self.conv_w = nn.Sequential(
			nn.Conv2d(knn_feats, hidden_feats, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_feats, n_feats, kernel_size=3, padding=1),
		)
		self.conv_b = nn.Sequential(
			nn.Conv2d(knn_feats, hidden_feats, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_feats, n_feats, kernel_size=3, padding=1),
		)

		self.conv_res = ResBlock(conv=conv, n_feats=n_feats, kernel_size=3)
		return
	
	def forward(self, x):
		# x[0]:feature; x[1]:shared knn
		x0, knn = x[0], x[1]

		w = F.leaky_relu(self.conv_w(knn), 0.1, inplace=True)
		b = F.leaky_relu(self.conv_b(knn), 0.1, inplace=True)

		x1 = x0 * w + b # affine transform
		x2 = self.conv_res(x0)
		return (x2+x0, knn)
		

class FusConcatLayer(nn.Module):
	def __init__(self, n_feats, knn_feats, hidden_feats=64, conv=default_conv):
		super(FusConcatLayer, self).__init__()

		self.conv_knn = nn.Sequential(
			nn.Conv2d(knn_feats, hidden_feats, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_feats, n_feats, kernel_size=3, padding=1),
		)

		self.conv = nn.Sequential(
			conv(2*n_feats, n_feats, kernel_size=3),
			nn.ReLU(inplace=True),
			ResBlock(conv, n_feats, kernel_size=3)
		)
		return

	def forward(self, x):
		# x[0]:feature; x[1]:shared knn
		x0, knn = x[0], x[1]

		k_feat = F.leaky_relu(self.conv_knn(knn), 0.1, inplace=True)
		# k_feat = self.conv_knn(knn)

		x1 = torch.cat((x0, k_feat), 1)	# concat
		x2 = self.conv(x1)

		return (x2 + x0, knn)