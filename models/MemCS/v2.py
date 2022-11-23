import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class MLP(nn.Module):
		def __init__(self, in_feats, hidden_feats=None, out_feats=None, act_layer=nn.GELU, drop=0.):
				super().__init__()
				hidden_feats = in_feats if hidden_feats is None else hidden_feats
				out_feats = in_feats if out_feats is None else out_feats

				self.fc1 = nn.Linear(in_feats, out_feats)
				self.act = act_layer()
				self.fc2 = nn.Linear(in_feats, out_feats)
				self.drop = nn.Dropout(drop)

		def forward(self, x):
				out = self.fc1(x)
				out = self.act(out)
				out = self.drop(out)
				out = self.fc2(out)
				out = self.drop(out)
				return out


class LePEAttention(nn.Module):
		def __init__(self, n_feats, resolution, idx, split_size, dim_out=None, num_heads=8, attn_drop=0., qk_scale=None):
				super().__init__()
				self.n_feats = n_feats
				self.resolution = resolution
				self.split_size = split_size
				self.num_heads = num_heads

				head_dim = n_feats // num_heads

				self.attn_type = 'dot'
				if self.attn_type == 'dot':
					self.scale = qk_scale or head_dim ** -0.5
				elif self.attn_type == 'l2':
					self.scale = qk_scale or head_dim ** -0.5

				if idx == 0:
						H_sp, W_sp = self.resolution, self.split_size
				elif idx == 1:
						W_sp, H_sp = self.resolution, self.split_size
				else:
						raise Exception('Unknown idx:{idx}')
				self.H_sp = H_sp
				self.W_sp = W_sp

				self.get_v = nn.Conv2d(
						n_feats, n_feats, kernel_size=3, stride=1, padding=1, groups=n_feats)
				self.attn_drop = nn.Dropout(attn_drop)

		def im2cswin(self, x):
				N, K, C = x.shape
				H = W = int(np.sqrt(K))
				x = x.transpose(-2, -1).contiguous().view(N, C, H, W)
				x = img2windows(x, self.H_sp, self.W_sp)
				x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads,
											C // self.num_heads).permute(0, 2, 1, 3).contiguous()
				return x

		def get_lepe(self, x, func):
				B, N, C = x.shape
				H = W = int(np.sqrt(N))
				x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

				H_sp, W_sp = self.H_sp, self.W_sp
				x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
				x = x.permute(0, 2, 4, 1, 3, 5).contiguous(
				).reshape(-1, C, H_sp, W_sp)  # B', C, H', W'

				lepe = func(x)  # B', C, H', W'
				lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads,
														H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

				x = x.reshape(-1, self.num_heads, C // self.num_heads,
											self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
				return x, lepe

		def forward(self, qkv, mask=None):
				_, N, K, C = qkv.shape
				H = W = self.resolution

				q, k, v = qkv[0], qkv[1], qkv[2]

				q = self.im2cswin(q)
				k = self.im2cswin(k)

				v, lepe = self.get_lepe(v, self.get_v)
				
				q = q * self.scale

				# attn
				if self.attn_type == 'dot':
					attn = (q @ k.transpose(-2, -1))	
				elif self.attn_type == 'l2':
					attn = self.l2_attn(q,k)
				
				attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
				attn = self.attn_drop(attn)

				x = (attn @ v) + lepe
				x = x.transpose(1, 2).reshape(-1, self.H_sp *
																			self.W_sp, C)  # B head N N @ B head N C

				# Window2Img
				x = windows2img(x, self.H_sp, self.W_sp, H,
												W).view(N, -1, C)  # N H' W' C

				return x

		def l2_attn(self, q, k):
			N, H, K,  C = q.shape
			dot = q @ k.transpose(-2, -1)

			q_mod = torch.sum(q*q, dim=-1).unsqueeze(-1).repeat(1,1,1,K)
			k_mod = torch.sum(k*k, dim=-1).unsqueeze(-1).repeat(1,1,1,K)

			# attn = 1.0 / torch.sqrt(q_mod + k_mod - dot)
			# attn = 1.0 - torch.sqrt(q_mod + k_mod - dot)
			attn = 0.5 - torch.sqrt(q_mod + k_mod - dot)
			return attn
		



class ChannelAttention(nn.Module):
		def __init__(self, n_feats, squeeze_factor):
				super().__init__()

				self.low_feats = n_feats // squeeze_factor

				self.pool = nn.AdaptiveAvgPool1d(1)

				self.down = nn.Sequential(
						nn.Linear(n_feats, n_feats // squeeze_factor)
				)
				self.up = nn.Sequential(
						nn.Linear(n_feats // squeeze_factor, n_feats),
						nn.Sigmoid()
				)

				# Memory Bank size can be changed [low_feats, memSize]
				# It seems that if memSize and loss_feats are equal, it will have better effect ? ? ?
				memSize = 256
				self.memBank = torch.nn.Parameter(
						torch.randn(n_feats // squeeze_factor, memSize))

		def forward(self, x):
				N, K, C = x.shape
				out = x.transpose(1, 2)  # [N, K, C]->[N, C, K]

				out = self.pool(out).squeeze(-1)  # [N, C, K]->[N, C]
				out_down = self.down(out).unsqueeze(2)  # [N, C]->[N, low_feats, 1]

				# mem
				# [low_feats, memSize]->[N, low_feats, memSize]
				mem = self.memBank.unsqueeze(0).repeat(N, 1, 1)
				# [N, 1, low_feats]×[N, low_feats, memSize]->[N, 1, mem_size]
				data = (out_down.transpose(1, 2)) @ mem
				# [N, 1, mem_size]
				data_soft = F.softmax(data * (int(self.low_feats) ** (-0.5)), dim=-1)
				# [N, 1, mem_size]×[N, memSize, low_feats]->[N, 1, low_feats]
				mem_select = data_soft @ mem.transpose(1, 2)

				out_up = self.up(mem_select)
				out = x * out_up
				return out


class CAB(nn.Module):
		def __init__(self, n_feats, compress_ratio, squeeze_factor):
				super(CAB, self).__init__()
				self.n_feats = n_feats
				self.cab = nn.Sequential(
						nn.Linear(n_feats, n_feats // compress_ratio),
						nn.GELU(),
						nn.Linear(n_feats // compress_ratio, n_feats),
						ChannelAttention(n_feats, squeeze_factor)
				)

		def forward(self, x):
				return self.cab(x)

		def flops(self, shape):
				flops = 0
				H, W = shape
				flops += self.n_feats * H * W
				return flops


class WindowAttention(nn.Module):
		def __init__(self, n_feats, window_size, num_head, qkv_bias, qk_scale, atten_drop=0., proj_drop=0., split_size=1):
				super().__init__()

				self.qkv = nn.Linear(n_feats, n_feats * 3, bias=qkv_bias)

				self.attns = nn.ModuleList([LePEAttention(n_feats=n_feats//2, resolution=window_size[0], idx=i, split_size=split_size,
																									num_heads=num_head//2, attn_drop=atten_drop, qk_scale=qk_scale) for i in range(2)])

				self.c_attns = CAB(n_feats=n_feats, compress_ratio=4, squeeze_factor=8)

				self.proj = nn.Linear(n_feats, n_feats)
				self.proj_drop = nn.Dropout(proj_drop)

		def forward(self, x, mask=None):
				N, K, C = x.shape
				# RA branch
				qkv = self.qkv(x).reshape(N, -1, 3, C).permute(2, 0, 1, 3)  # [3, N, K, C]
				out1 = self.attns[0](qkv[:, :, :, :C//2], mask)
				out2 = self.attns[1](qkv[:, :, :, :C//2], mask)
				out3 = torch.cat([out1, out2], dim=2)
				out4 = rearrange(out3, 'b n (g d) -> b n ( d g)', g=4)
				# CA branch
				out5 = self.c_attns(x)
				# together
				out6 = out4 + 0.1 * out5
				out7 = self.proj(out6)
				out8 = self.proj_drop(out7)
				return out8


class SSMTDA(nn.Module):
		def __init__(self, n_feats, input_resolution, window_size, n_head, shift_size, split_size, drop_path, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, act_layer=nn.GELU):
				super().__init__()

				self.shift_size = shift_size
				self.window_size = window_size

				self.norm1 = nn.LayerNorm(n_feats)
				self.norm2 = nn.LayerNorm(n_feats)
				self.drop_path = DropPath(
						drop_path) if drop_path > 0. else nn.Identity()

				mlp_hidden_dim = int(n_feats * mlp_ratio)
				self.mlp = MLP(in_feats=n_feats, hidden_feats=mlp_hidden_dim,
											 out_feats=n_feats, act_layer=act_layer, drop=drop)

				self.attentions = WindowAttention(
						n_feats=n_feats, window_size=to_2tuple(self.window_size), num_head=n_head, qkv_bias=qkv_bias, qk_scale=qk_scale, atten_drop=attn_drop, proj_drop=drop, split_size=split_size)

		def forward(self, x):
				N, C, H, W = x.shape
				out0 = x.flatten(2).transpose(1, 2)  # [N, C, H, W]->[N, H*W, C]

				out1 = self.norm1(out0).view(N, H, W, C)

				# cyclic window
				if self.shift_size > 0:
						out2 = torch.roll(
								out1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
				else:
						out2 = out1

				out3 = window_partition(out2, self.window_size)
				out4 = out3.view(-1, self.window_size * self.window_size, C)

				out5 = self.attentions(out4)

				out6 = out5.view(-1, self.window_size, self.window_size, C)
				out7 = window_reverse(out6, self.window_size, H, W)

				# shift window
				if self.shift_size > 0:
						out8 = torch.roll(
								out7, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
				else:
						out8 = out7

				out9 = out8.view(N, H*W, C)

				# FFN
				out10 = self.drop_path(out9) + out0
				out11 = out10 + self.drop_path(self.mlp(self.norm2(out10)))

				out12 = out11.transpose(1, 2).view(N, C, H, W)
				return out12


class SMSBlock(nn.Module):
	def __init__(self, n_feats, window_size, n_layer, n_head, mlp_ratio, qkv_bias, qk_scale, drop_path, split_size):
		super().__init__()

		blocks = []
		for i in range(n_layer):
				blocks.append(
					SSMTDA(n_feats=n_feats, input_resolution=window_size, window_size=window_size, n_head=n_head, shift_size=0 if i % 2 == 0 else window_size//2,
								 split_size=split_size, drop_path=drop_path[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0., act_layer=nn.GELU)
				)
		self.blocks = nn.Sequential(*blocks)
		self.conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

	def forward(self, x):
		out = self.blocks(x)
		out = self.conv(out) + x
		return out


class MemCSV2(nn.Module):
		def __init__(self, channels, scale_factor):
				super().__init__()

				n_feats = 252

				win_size = [8, 8, 8, 8, 8, 8]

				n_layers = [6, 6, 6, 6, 6, 6]
				n_heads = [6, 6, 6, 6, 6, 6]

				split_sizes = [1, 1, 1, 1, 1, 1]
				mlp_ratio = 2
				qkv_bias = True
				qk_scale = None
				bias = False,
				drop_path_rate = 0.1

				self.conv_shallow = nn.Conv2d(
						channels, n_feats, kernel_size=3, stride=1, padding=1)

				dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(n_layers))]

				self.layers = nn.ModuleList()
				for i in range(len(n_layers)):
						layer = SMSBlock(n_feats=n_feats, window_size=win_size[i], n_layer=n_layers[i], n_head=n_heads[i], mlp_ratio=mlp_ratio,
														 qkv_bias=qkv_bias, qk_scale=qkv_bias, drop_path=dpr[sum(n_layers[:i]):sum(n_layers[:i + 1])], split_size=1)
						self.layers.append(layer)

				self.up = Upsampler(
						conv=default_conv,
						scale=scale_factor,
						n_feats=n_feats
				)

				self.conv_tail = nn.Conv2d(
						n_feats, channels, kernel_size=3, stride=1, padding=1)

		def forward(self, x):
				out1 = self.conv_shallow(x)

				out_i = out1
				for layer in self.layers:
						out_i = layer(out_i)

				out2 = self.up(out_i + out1)
				out3 = self.conv_tail(out2)
				return out3


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
		def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
				m = []
				if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
						for _ in range(int(math.log(scale, 2))):
								m.append(conv(n_feats, 4 * n_feats, 3, bias))
								m.append(nn.PixelShuffle(2))
								if bn:
										m.append(nn.BatchNorm2d(n_feats))
								if act == 'relu':
										m.append(nn.ReLU(True))
								elif act == 'prelu':
										m.append(nn.PReLU(n_feats))
				elif scale == 3:
						m.append(conv(n_feats, 9 * n_feats, 3, bias))
						m.append(nn.PixelShuffle(3))
						if bn:
								m.append(nn.BatchNorm2d(n_feats))
						if act == 'relu':
								m.append(nn.ReLU(True))
						elif act == 'prelu':
								m.append(nn.PReLU(n_feats))
				else:
						raise NotImplementedError

				super(Upsampler, self).__init__(*m)
