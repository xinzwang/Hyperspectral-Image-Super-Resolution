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
	else:
		raise Exception('Unknown dataset:', name)

def build_dataset(dataset, path, batch_size=32, scale_factor=2, test_flag=False):
	datasetObj = SelectDatasetObject(dataset)
	dataset = datasetObj(
		path=path,
		scale_factor=scale_factor, 
		test_flag=test_flag,
	)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size if not test_flag else 1,
		num_workers=8,
		shuffle= (not test_flag)	# shuffle only train
	)
	return dataset, dataloader