"""
Common files
"""
import torchvision.transforms as transforms



def build_transforms(opts):
	trans = []
	for opt in opts:
		if opt['name'] == 'ToTensor':
			trans.append(transforms.ToTensor())	# HWC->CHW; [0, 255]->[0, 1]; np.ndarray->torch.FloatTensor
		elif opt['name'] == 'CenterCrop':
			trans.append(transforms.CenterCrop(opt['size']))
		elif opt['name'] == 'RandomRotation'
			trans.append(transforms.RandomRotation())
		else:
			raise Exception('Unknown opt:', opt)
	return transforms.Compose(trans)


