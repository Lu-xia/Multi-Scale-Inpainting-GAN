import cupy as cp
import chainer.functions as F
import numpy as np


def gradient_loss(generated, truth):
	"""
	:param generated: generated image by the generator at any scale
	:param truth: The ground truth image at that scale
	:return: GDL loss
	"""
	xp = cp.get_array_module(generated.data)
	n, c, h, w = generated.shape
	wx = xp.array([[[1, -1]]]*c, ndmin=4).astype(xp.float32)
	wy = xp.array([[[1], [-1]]]*c, ndmin=4).astype(xp.float32)

	d_gx = nn.torch(generated, wx)
	d_gy = F.convolution_2d(generated, wy)

	d_tx = F.convolution_2d(truth, wx)
	d_ty = F.convolution_2d(truth, wy)

	return (F.sum(F.absolute(d_gx - d_tx)) + F.sum(F.absolute(d_gy - d_ty)))


