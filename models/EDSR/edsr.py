import torch.nn as nn
from models.EDSR import common

class EDSR(nn.Module):
	def __init__(self, channels, scale_factor):
				super(EDSR, self).__init__()
				
				n_colors = channels
				scale = scale_factor

				n_resblocks = 16
				n_feats = 64
				res_scale = 1
				kernel_size = 3
				
				act = nn.ReLU(True)

				conv=common.default_conv

				# url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
				# if url_name in url:
				# 		self.url = url[url_name]
				# else:
				# 		self.url = None
				# self.sub_mean = common.MeanShift(rgb_range)
				# self.add_mean = common.MeanShift(rgb_range, sign=1)

				# define head module
				m_head = [conv(n_colors, n_feats, kernel_size)]

				# define body module
				m_body = [
						common.ResBlock(
								conv, n_feats, kernel_size, act=act, res_scale=res_scale
						) for _ in range(n_resblocks)
				]
				m_body.append(conv(n_feats, n_feats, kernel_size))

				# define tail module
				m_tail = [
						common.Upsampler(conv, scale, n_feats, act=False),
						conv(n_feats, n_colors, kernel_size)
				]

				self.head = nn.Sequential(*m_head)
				self.body = nn.Sequential(*m_body)
				self.tail = nn.Sequential(*m_tail)

	def forward(self, x):
			# x = self.sub_mean(x)
			x = self.head(x)
			res = self.body(x)
			res += x
			x = self.tail(res)
			# x = self.add_mean(x)
			return x 

	def load_state_dict(self, state_dict, strict=True):
			own_state = self.state_dict()
			for name, param in state_dict.items():
					if name in own_state:
							if isinstance(param, nn.Parameter):
									param = param.data
							try:
									own_state[name].copy_(param)
							except Exception:
									if name.find('tail') == -1:
											raise RuntimeError('While copying the parameter named {}, '
																				 'whose dimensions in the model are {} and '
																				 'whose dimensions in the checkpoint are {}.'
																				 .format(name, own_state[name].size(), param.size()))
					elif strict:
							if name.find('tail') == -1:
									raise KeyError('unexpected key "{}" in state_dict'
																 .format(name))