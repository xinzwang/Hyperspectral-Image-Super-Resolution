import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


band_means = {
	'CAVE': (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
					 0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
					 0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541), #CAVE

	'DIV2K': (0.4488, 0.4371, 0.4040)	# DIV2K
}

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


class BasicConv3d(nn.Module):
		def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
				super(BasicConv3d, self).__init__()
				self.conv = wn(nn.Conv3d(in_channel, out_channel,
															kernel_size=kernel_size, stride=stride,
															padding=padding))
				# self.relu = nn.ReLU(inplace=True)

		def forward(self, x):
				x = self.conv(x)
				# x = self.relu(x)
				return x


class Block3D(nn.Module):
	def __init__(self, wn, n_feats):
		super(Block3D, self).__init__()
		self.conv = nn.Sequential(
				BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
				nn.ReLU(inplace=True),
				BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
		)            
			 
	def forward(self, x): 
		return self.conv(x)


class ResBlock3D(nn.Module):
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.conv = nn.Sequential(
			nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1)),
			act,
			nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1)),
		)
	
	def forward(self, x):
		# x: [N, F, C, H, W]
		N,F,C,H,W = x.shape
		out = x.transpose(1,2).reshape(N*C, F, H, W) # [N*C, F, H, W]

		return out.reshape(N,C,F,H,W).transpose(1,2)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CALayer, self).__init__()
		# global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# feature channel downscale and upscale --> channel weight
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y


class SpectralSpatialCALayer(nn.Module):
	def __init__(self, channels, n_feats, cfgs):
		super().__init__()
		self.spec_pool = nn.AdaptiveAgvPool2d()

	def max_avg_pool(self, x, dim):
		return torch.cat((torch.max(x, dim)[0].unsqueeze(1), torch.mean(x, dim).unsqueeze(1)), dim=1)	# [max_pool, avg_pool];

	def forward(self, x, rgb_mode=False):
		N, F, C, H, W = x.shape

		band_attn = self.max_avg_pool(x, dim=1)	# [N, 2, C, H, W]

		spec_attn = self.max_avg_pool(x, dim=1) # [N, 2, C, H, W]

		spec_attn = self.max_avg_pool(x) #[N, F, 2, H, W]



class CALayer2DFeatsDim(nn.Module):
	def __init__(self, channels, n_feats, reduction=1):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.conv_du = nn.Sequential(
			nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(1, 1), padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(n_feats // reduction, n_feats, kernel_size=(1, 1), padding=0, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F, C, H, W = x.shape
		out = x.transpose(1,2).reshape(N * C, F, H, W) # [N, C, F, H, W]

		attn = self.pool(out)
		attn = self.conv_du(attn)
		out = out * attn

		out = out.reshape(N, C, F, H, W).transpose(1, 2)	# [N, F, C, H, W]
		return out


class CALayer2DFeatsDimWithChansPosEmbed(nn.Module):
	def __init__(self, channels, n_feats, reduction=4, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.conv_1 = nn.Sequential(
			nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(1, 1), stride=(1, 1)),
			act,
		)

		self.conv_2 = nn.Sequential(
			nn.Conv2d(n_feats // reduction, n_feats, kernel_size=(1, 1), stride=(1, 1)),
			nn.Sigmoid()
		)

		self.chan_pos_embed = nn.Parameter(torch.randn(channels, n_feats, 1, 1))	# [C, F, 1, 1]
		self.rgb_pos_embed = nn.Parameter(torch.randn(3, n_feats, 1, 1))	# [C, F, 1, 1]

	def forward(self, x, rgb_mode=False):
		# x: [N, F, C, H, W]
		N, F, C, H, W = x.shape
		out = x.transpose(1,2).reshape(N * C, F, H, W) # [N*C, F, H, W]

		attn = self.pool(out)	# [N*C, F, 1, 1]
		attn = self.conv_1(attn)
		attn = self.conv_2(attn) + (self.rgb_pos_embed if rgb_mode else self.chan_pos_embed).repeat(N, 1, 1, 1)
		out = out * attn

		out = out.reshape(N, C, F, H, W).transpose(1, 2)	# [N, F, C, H, W]
		return out

class BandSALayer3D(nn.Module):
	def __init__(self, channels, n_feats, cfgs, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		n_heads = cfgs['n_heads']
		attn_drop = cfgs['attn_drop']
		proj_drop = cfgs['proj_drop']

		self.n_heads = n_heads

		self.pool = nn.AdaptiveAvgPool2d((1, 1))

		self.qk = nn.Linear(n_feats , 2 * n_feats)
		self.v = nn.Conv3d(n_feats, n_feats, kernel_size=(1, 1, 1))
		self.softmax = nn.Softmax(dim=-1)
		self.scale = n_heads ** -0.5
		self.attn_drop = nn.Dropout(attn_drop)

		self.proj = nn.Conv3d(n_feats, n_feats, kernel_size=(1, 1, 1), padding=(0, 0, 0))
		self.proj_drop = nn.Dropout(proj_drop)

		self.conv = nn.Conv3d(2*n_feats, n_feats, kernel_size=(1, 1, 1), padding=(0,0,0))


	def forward(self, x, rgb_mode=False):
		# x: [N, F, C, H, W]
		N, F, C, H, W = x.shape
		out = x.transpose(1,2) # [N, C, F, H, W]

		# adaptive avg pool with Band pos_embed
		pool  = self.pool(out)	# [N, C,  F, 1, 1]
		pool = pool.reshape(N*self.channels, self.n_feats)
		
		# self-attention
		qk = self.qk(pool).reshape(N, C, 2, self.n_heads, F // self.n_heads).permute(2, 0, 3, 1, 4)	# [qkv, N, n_heads,  C, F//n_heads]
		q, k = qk[0], qk[1]	# [N, n_heads,  C, F//n_heads]
		v = self.v(x).reshape(N, self.n_heads, F // self.n_heads, C, H, W).permute(0, 1, 4, 5, 3, 2)	# [N, n_heads, H, W, C, F//heads]

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))	# [N, num_heads, C, C]

		attn = self.softmax(attn)	
		attn = self.attn_drop(attn)	# [N, num_heads, C, C]

		val = (attn.unsqueeze(2).unsqueeze(2) @ v).permute(0, 1, 5, 4, 2, 3).reshape(N, F, C, H, W)
		# val = torch.cat((x, val), dim=1)
		out = self.proj_drop(self.proj(val))

		out = torch.cat((x, val), dim=1)
		out = self.conv(out)

		return out

# class BandSALayer3D(nn.Module):
# 	def __init__(self, channels, n_feats, cfgs, act=nn.ReLU(inplace=True)):
# 		super().__init__()
# 		self.channels = channels
# 		self.n_feats = n_feats

# 		n_heads = cfgs['n_heads']
# 		attn_dims = cfgs['attn_dims']
# 		attn_drop = cfgs['attn_drop']
# 		proj_drop = cfgs['proj_drop']

# 		self.n_heads = n_heads
# 		self.attn_dims = attn_dims

# 		self.pool = nn.AdaptiveAvgPool2d((1, 1))

# 		self.qk = nn.Linear(n_feats , 2 * (attn_dims * n_heads))
# 		self.v = nn.Conv3d(n_feats, attn_dims * n_heads, kernel_size=(1, 1, 1))
# 		self.softmax = nn.Softmax(dim=-1)
# 		self.scale = n_heads ** -0.5
# 		self.attn_drop = nn.Dropout(attn_drop)

# 		# self.proj = nn.Linear(n_feats, n_feats)
# 		self.proj = nn.Conv3d(attn_dims * n_heads, n_feats, kernel_size=(1, 1, 1))
# 		self.proj_drop = nn.Dropout(proj_drop)

# 	def forward(self, x):
# 		# x: [N, F, C, H, W]
# 		N, F, C, H, W = x.shape
# 		out = x.transpose(1,2) # [N, C, F, H, W]

# 		# adaptive avg pool with Band pos_embed
# 		pool  = self.pool(out)	# [N, C,  F, 1, 1]
# 		pool = pool.reshape(N*self.channels, self.n_feats)
		
# 		# self-attention
# 		qk = self.qk(pool).reshape(N, C, 2, self.n_heads, self.attn_dims).permute(2, 0, 3, 1, 4)	# [qkv, N, n_heads,  C, dims]
# 		q, k = qk[0], qk[1]	# [N, n_heads,  C, F//n_heads]
# 		v = self.v(x).reshape(N, self.n_heads, self.attn_dims, C, H, W).permute(0, 1, 4, 5, 3, 2)	# [N, n_heads, H, W, C, dims]

# 		q = q * self.scale
# 		attn = (q @ k.transpose(-2, -1))	# [N, num_heads, C, C]

# 		attn = self.softmax(attn)
# 		attn = self.attn_drop(attn)	# [N, num_heads, C, C]

# 		val = (attn.unsqueeze(2).unsqueeze(2) @ v).permute(0, 1, 5, 4, 2, 3).reshape(N, self.n_heads*self.attn_dims, C, H, W)
# 		val = self.proj_drop(self.proj(val))
# 		return val


class CALayer3DChannelDim(nn.Module):
	def __init__(self, channels, n_feats, reduction=4, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d((1, 1))

		self.fc = nn.Sequential(
			nn.Linear(n_feats * channels, n_feats * channels // reduction),
			act,
			nn.Linear(n_feats * channels // reduction, n_feats * channels),
			nn.Sigmoid(),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		attn  = self.pool(x).reshape(-1, self.n_feats * self.channels )	# [N, F*C]
		attn = self.fc(attn).reshape(-1, self.n_feats, self.channels).unsqueeze(-1).unsqueeze(-1)
		out = x * attn
		return out


class DoubleConv2DFeatsDim(nn.Module):
	"""从特定波段，从不同的特征维度，提取空间特征。空间维度共享参数，光谱维度共享参数，特征维度独立参数。"""
	def __init__(self, channels, n_feats, kernel_size=3, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv2d(n_feats, n_feats, kernel_size=(kernel_size), stride=1, padding=(kernel_size//2, kernel_size//2)),
			act,
			nn.Conv2d(n_feats, n_feats, kernel_size=(kernel_size), stride=1, padding=(kernel_size//2, kernel_size//2)),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F,C,H,W = x.shape
		out = x.transpose(1, 2).reshape(N*C, F, H, W)
		out = self.conv(out)
		return out.reshape(N, C, F, H, W).transpose(1, 2)


class DoubleConv3DFeatsDim(nn.Module):
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)),
			act,
			nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		out = x.transpose(1, 2)	# [N, C, F, H, W]
		out = self.conv(out)
		return out.transpose(1, 2)


class DoubleConv2DChannelDim(nn.Module):
	"""效果不好"""
	"""从所有波段，特定的特征维度，提取空间、光谱维度信息。空间维度共享参数，光谱维度独立参数，特征维度共享参数"""
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=(1,1)),
			act,
			nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=(1,1)),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F,C,H,W = x.shape
		out = x.reshape(N*F, C, H, W)
		out = self.conv(out)
		return out.reshape(N, F, C, H, W)


class DoubleConv3DChannelDim(nn.Module):
	"""从所有波段，所有的特征维度，提取空间、光谱维度信息。 空间维度共享参数，光谱维度共享参数，特征维度独立参数"""
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)),
			act,
			nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)),
		)

	def forward(self, x):
		out = self.conv(x)
		return out


"""
######################  Swin Transformer Tools  ######################
"""
class Mlp(nn.Module):
		def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
				super().__init__()
				out_features = out_features or in_features
				hidden_features = hidden_features or in_features
				self.fc1 = nn.Linear(in_features, hidden_features)
				self.act = act_layer()
				self.fc2 = nn.Linear(hidden_features, out_features)
				self.drop = nn.Dropout(drop)

		def forward(self, x):
				x = self.fc1(x)
				x = self.act(x)
				x = self.drop(x)
				x = self.fc2(x)
				x = self.drop(x)
				return x


def window_partition(x, window_size):
		"""
		Args:
				x: (B, H, W, C)
				window_size (int): window size
		Returns:
				windows: (num_windows*B, window_size, window_size, C)
		"""
		B, H, W, C = x.shape 	# [1, 64, 64, 1]
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


class WindowAttention(nn.Module):
		r""" Window based multi-head self attention (W-MSA) module with relative position bias.
		It supports both of shifted and non-shifted window.
		Args:
				dim (int): Number of input channels.
				window_size (tuple[int]): The height and width of the window.
				num_heads (int): Number of attention heads.
				qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
				qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
				attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
				proj_drop (float, optional): Dropout ratio of output. Default: 0.0
		"""

		def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

				super().__init__()
				self.dim = dim
				self.window_size = window_size  # Wh, Ww
				self.num_heads = num_heads
				head_dim = dim // num_heads
				self.scale = qk_scale or head_dim ** -0.5

				# define a parameter table of relative position bias
				self.relative_position_bias_table = nn.Parameter(
						torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

				# get pair-wise relative position index for each token inside the window
				coords_h = torch.arange(self.window_size[0])
				coords_w = torch.arange(self.window_size[1])
				coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
				coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
				relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
				relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
				relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
				relative_coords[:, :, 1] += self.window_size[1] - 1
				relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
				relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
				self.register_buffer("relative_position_index", relative_position_index)

				self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
				self.attn_drop = nn.Dropout(attn_drop)
				self.proj = nn.Linear(dim, dim)

				self.proj_drop = nn.Dropout(proj_drop)

				trunc_normal_(self.relative_position_bias_table, std=.02)
				self.softmax = nn.Softmax(dim=-1)

		def forward(self, x, mask=None):
				"""
				Args:
						x: input features with shape of (num_windows*B, N, C)
						mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
				"""
				B_, N, C = x.shape
				qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
				q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

				q = q * self.scale
				attn = (q @ k.transpose(-2, -1))

				relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
						self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
				relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
				attn = attn + relative_position_bias.unsqueeze(0)

				if mask is not None:
						nW = mask.shape[0]
						attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
						attn = attn.view(-1, self.num_heads, N, N)
						attn = self.softmax(attn)
				else:
						attn = self.softmax(attn)

				attn = self.attn_drop(attn)

				x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
				x = self.proj(x)
				x = self.proj_drop(x)
				return x

		def extra_repr(self) -> str:
				return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

		def flops(self, N):
				# calculate flops for 1 window with token length of N
				flops = 0
				# qkv = self.qkv(x)
				flops += N * self.dim * 3 * self.dim
				# attn = (q @ k.transpose(-2, -1))
				flops += self.num_heads * N * (self.dim // self.num_heads) * N
				#  x = (attn @ v)
				flops += self.num_heads * N * N * (self.dim // self.num_heads)
				# x = self.proj(x)
				flops += N * self.dim * self.dim
				return flops


class PatchEmbed(nn.Module):
		r""" Image to Patch Embedding
		Args:
				img_size (int): Image size.  Default: 224.
				patch_size (int): Patch token size. Default: 4.
				in_chans (int): Number of input image channels. Default: 3.
				embed_dim (int): Number of linear projection output channels. Default: 96.
				norm_layer (nn.Module, optional): Normalization layer. Default: None
		"""

		def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
				super().__init__()
				img_size = to_2tuple(img_size)
				patch_size = to_2tuple(patch_size)
				patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
				self.img_size = img_size
				self.patch_size = patch_size
				self.patches_resolution = patches_resolution
				self.num_patches = patches_resolution[0] * patches_resolution[1]

				self.in_chans = in_chans
				self.embed_dim = embed_dim

				if norm_layer is not None:
						self.norm = norm_layer(embed_dim)
				else:
						self.norm = None

		def forward(self, x):
				x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C		# [N, H*W, C]
				if self.norm is not None:
						x = self.norm(x)
				return x

		def flops(self):
				flops = 0
				H, W = self.img_size
				if self.norm is not None:
						flops += H * W * self.embed_dim
				return flops


class PatchUnEmbed(nn.Module):
		r""" Image to Patch Unembedding
		Args:
				img_size (int): Image size.  Default: 224.
				patch_size (int): Patch token size. Default: 4.
				in_chans (int): Number of input image channels. Default: 3.
				embed_dim (int): Number of linear projection output channels. Default: 96.
				norm_layer (nn.Module, optional): Normalization layer. Default: None
		"""

		def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
				super().__init__()
				img_size = to_2tuple(img_size)
				patch_size = to_2tuple(patch_size)
				patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
				self.img_size = img_size
				self.patch_size = patch_size
				self.patches_resolution = patches_resolution
				self.num_patches = patches_resolution[0] * patches_resolution[1]

				self.in_chans = in_chans
				self.embed_dim = embed_dim

		def forward(self, x, x_size):
				B, HW, C = x.shape
				x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
				return x

		def flops(self):
				flops = 0
				return flops


class SwinTransformerBlock(nn.Module):
	r""" Swin Transformer Block.
		Args:
				dim (int): Number of input channels.
				input_resolution (tuple[int]): Input resulotion.
				num_heads (int): Number of attention heads.
				window_size (int): Window size.
				shift_size (int): Shift size for SW-MSA.
				mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
				qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
				qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
				drop (float, optional): Dropout rate. Default: 0.0
				attn_drop (float, optional): Attention dropout rate. Default: 0.0
				drop_path (float, optional): Stochastic depth rate. Default: 0.0
				act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
				norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
		"""

	def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
								 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
								 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio
		if min(self.input_resolution) <= self.window_size:
				# if window size is larger than input resolution, we don't partition windows
				self.shift_size = 0
				self.window_size = min(self.input_resolution)
		assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention(
				dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
				qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

		if self.shift_size > 0:
				attn_mask = self.calculate_mask(self.input_resolution)
		else:
				attn_mask = None

		self.register_buffer("attn_mask", attn_mask)

	def calculate_mask(self, x_size):
		# calculate attention mask for SW-MSA
		H, W = x_size
		img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
		h_slices = (slice(0, -self.window_size),
								slice(-self.window_size, -self.shift_size),
								slice(-self.shift_size, None))
		w_slices = (slice(0, -self.window_size),
								slice(-self.window_size, -self.shift_size),
								slice(-self.shift_size, None))
		cnt = 0
		for h in h_slices:
				for w in w_slices:
						img_mask[:, h, w, :] = cnt
						cnt += 1

		mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
		mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
		attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
		attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

		return attn_mask

	def forward(self, x, x_size):
		H, W = x_size
		B, L, C = x.shape
		# assert L == H * W, "input feature has wrong size"

		shortcut = x
		x = self.norm1(x)
		x = x.view(B, H, W, C)

		# cyclic shift
		if self.shift_size > 0:
				shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
		else:
				shifted_x = x

		# partition windows
		x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
		x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

		# W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
		if self.input_resolution == x_size:
				attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
		else:
				attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

		# merge windows
		attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
		shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

		# reverse cyclic shift
		if self.shift_size > 0:
				x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
				x = shifted_x
		x = x.view(B, H * W, C)

		# FFN
		x = shortcut + self.drop_path(x)
		x = x + self.drop_path(self.mlp(self.norm2(x)))

		return x

	def extra_repr(self) -> str:
		return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
					 f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

	def flops(self):
		flops = 0
		H, W = self.input_resolution
		# norm1
		flops += self.dim * H * W
		# W-MSA/SW-MSA
		nW = H * W / self.window_size / self.window_size
		flops += nW * self.attn.flops(self.window_size * self.window_size)
		# mlp
		flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
		# norm2
		flops += self.dim * H * W
		return flops



"""
######################  Upsampler tools  ######################
"""
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


