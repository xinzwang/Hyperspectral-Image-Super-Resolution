import logging

import torch
from torch.utils.data.dataloader import DataLoader
from datasets import *


# seed
def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return


# logger
def create_logger(log_path):
	logger = logging.getLogger(__name__)
	logger.setLevel(level = logging.INFO)
	handler = logging.FileHandler(log_path)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger


# datasets
def SelectDatasetObject(name):
	if name in ['Pavia', 'PaviaU', 'Salinas', 'KSC', 'Indian', 'CAVE']:
		return SingleDataset
	elif name in ['ICVL']:
		return MultiDataset
	elif name in ['DIV2K']:
		return DIV2KDataset
	elif name in ['CAVE_rgb']:
		return HSI_RGB
	else:
		raise Exception('Unknown dataset:', name)

def build_dataset(name, scale_factor, state='test', use_lms=False, batch_size=16):
	data_root = '/data2/wangxinzhe/codes/datasets/'
	test_flag = (state in ['test', 'val'])
	if name == 'CAVE':
		hr_path = data_root + f'CAVE/{state}/HR/'
		lr_path = data_root + f'CAVE/{state}/LR/X{scale_factor}/'
		lms_path = data_root + f'CAVE/{state}/LMS/X{scale_factor}/'
		dataset = HSIDataset(lr_path=lr_path, hr_path=hr_path, lms_path=lms_path if use_lms else None,
												 test_flag=test_flag,cache_ram=False,use_lms=use_lms)
	elif name == 'Harvard':
		hr_path = data_root + f'Harvard/{state}/HR/'
		lr_path = data_root + f'Harvard/{state}/LR/X{scale_factor}/'
		lms_path = data_root + f'Harvard/{state}/LMS/X{scale_factor}/'
		dataset = HSIDataset(lr_path=lr_path, hr_path=hr_path, lms_path=lms_path if use_lms else None,
												 test_flag=test_flag,cache_ram=True,use_lms=use_lms)
	else:
		raise Exception('Unknown dataset name:', name)
	dataloader = DataLoader(dataset=dataset, batch_size=batch_size if not test_flag else 1,
													pin_memory=True, shuffle=(not test_flag))
	return dataset, dataloader


def GetChansNumByDataName(name):
	if name in ['CAVE', 'ICVL', 'Harvard']:
		channels = 31
	else:
		raise Exception('Unknown Dataset:', name)
	return channels