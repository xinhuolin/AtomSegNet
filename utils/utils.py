import torch
from torchvision.transforms import ToTensor
from PyQt5 import QtGui
import os
import numpy as np


def GetIndexRangeOfBlk(height, width, blk_row, blk_col, blk_r, blk_c, over_lap = 0):
	blk_h_size = height//blk_row
	blk_w_size = width//blk_col

	if blk_r >= blk_row or blk_c >= blk_col:
		raise Exception("index is out of range...")

	upper_left_r = blk_r * blk_h_size
	upper_left_c = blk_c * blk_w_size
	ol_upper_left_r = max(upper_left_r - over_lap, 0)
	ol_upper_left_c = max(upper_left_c - over_lap, 0)

	if blk_r == (blk_row - 1):
		lower_right_r = height
		ol_lower_right_r = lower_right_r
	else:
		lower_right_r = upper_left_r + blk_h_size
		ol_lower_right_r = min(lower_right_r + over_lap, height)

	if blk_c == (blk_col - 1):
		lower_right_c = width
		ol_lower_right_c = lower_right_c
	else:
		lower_right_c = upper_left_c + blk_w_size
		ol_lower_right_c = min(lower_right_c + over_lap, width)

	return (upper_left_c, upper_left_r, lower_right_c, lower_right_r), (ol_upper_left_c, ol_upper_left_r, ol_lower_right_c, ol_lower_right_r)


"""circularMask_mse_beta : /home/student/Documents/u-net-pytorch-original/lr001_weightdecay00001/"""
"""denoise&airysuperrez_beta : /home/student/Documents/u-net_denoising/dataset_small_mask/"""
"""circularMask_chi10_beta : /home/student/Documents/Atom Segmentation APP/AtomSegGUI/atomseg_bupt_new_10/"""
"""circularMask_chi100_beta : /home/student/Documents/Atom Segmentation APP/AtomSegGUI/atomseg_bupt_new_100/"""
"""gaussianMask+ : /home/student/Documents/Atom Segmentation APP/AtomSegGUI/atom_seg_gaussian_mask/"""


def load_model(model_path, data, cuda, iter = 1):
	if os.path.basename(model_path) == "Gen1-noNoiseNoBackgroundSuperresolution.pth":
		from mypackage.model.unet_standard import NestedUNet
		net = NestedUNet()
		if cuda:
			net = net.cuda()
		if cuda:
			net = torch.nn.DataParallel(net)
			net.load_state_dict(torch.load(model_path))
		else:
			net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})

		transform = ToTensor()
		ori_tensor = transform(data)
		ori_tensor = torch.unsqueeze(ori_tensor, 0)

		padding_left = 0
		padding_right = 0
		padding_top = 0
		padding_bottom = 0
		ori_height = ori_tensor.size()[2]
		ori_width = ori_tensor.size()[3]
		use_padding = False
		if ori_width >= ori_height:
			padsize = ori_width
		else:
			padsize = ori_height
		if np.log2(padsize) >= 7.0:
			if (np.log2(padsize) - 7) % 1 > 0:
				padsize = 2 ** (np.log2(padsize) // 1 + 1)
		else:
			padsize = 2 ** 7
		if ori_height < padsize:
			padding_top = int(padsize - ori_height) // 2
			padding_bottom = int(padsize - ori_height - padding_top)
			use_padding = True
		if ori_width < padsize:
			padding_left = int(padsize - ori_width) // 2
			padding_right = int(padsize - ori_width - padding_left)
			use_padding = True
		if use_padding:
			padding_transform = torch.nn.ConstantPad2d((padding_left, \
														padding_right, \
														padding_top, \
														padding_bottom), 0)
			ori_tensor = padding_transform(ori_tensor)

		if cuda:
			ori_tensor = ori_tensor.cuda()
		output = net(ori_tensor)

		if cuda:
			result = (output.data).cpu().numpy()
		else:
			result = (output.data).numpy()

		padsize = int(padsize)
		result = result[0, 0, padding_top:(padsize - padding_bottom),
				 padding_left:(padsize - padding_right)]
	else:
		from model_structure.unet_sigmoid import UNet

		use_padding = False
		unet = UNet()
		unet = unet.eval()

		if cuda:
			unet = unet.cuda()

		if not cuda:
			unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
		else:
			unet.load_state_dict(torch.load(model_path))

		#transform = ToTensor()
		ori_tensor = ToTensor()(data)
		if cuda:
			#ori_tensor = Variable(ori_tensor.cuda())
			ori_tensor = ori_tensor.cuda()
		#else:
			#ori_tensor = Variable(ori_tensor)
			#ori_tensor = ori_tensor.cuda()
		ori_tensor = torch.unsqueeze(ori_tensor,0)

		padding_left = 0
		padding_right = 0
		padding_top = 0
		padding_bottom = 0
		ori_height = ori_tensor.size()[2]
		ori_width = ori_tensor.size()[3]

		if ori_height % 4:
			padding_top = (4 - ori_height % 4)//2
			padding_bottom = 4 - ori_height % 4 - padding_top
			use_padding = True
		if ori_width % 4:
			padding_left = (4 - ori_width % 4)//2
			padding_right = 4 - ori_width % 4 - padding_left
			use_padding = True
		if use_padding:
			padding_transform = torch.nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
			ori_tensor = padding_transform(ori_tensor)

		output = ori_tensor
		with torch.no_grad():
			for _ in range(iter):
				output = unet(output)

		if use_padding:
			output = output[:,:,padding_top : (padding_top + ori_height), padding_left : (padding_left + ori_width)]
		#
		# if cuda:
		# 	result = (output.data).cpu().numpy()
		# else:
		# 	result = (output.data).numpy()
		result = (output.data).cpu().numpy()

		result = result[0,0,:,:]
	return result


def PIL2Pixmap(im):
    """Convert PIL image to QImage """
    if im.mode == "RGB":
        pass
    elif im.mode == "L":
        im = im.convert("RGBA")
    data = im.convert("RGBA").tobytes("raw", "RGBA")
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qim)
    return pixmap

def map01(mat):
    return (mat - mat.min())/(mat.max() - mat.min())
