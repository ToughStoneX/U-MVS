# -*- coding: utf-8 -*-
# @Time    : 2020/09/08 17:05
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : sep_transforms
# @Software: PyCharm

import numpy as np
import torch
# from scipy.misc import imresize
from skimage.transform import resize as imresize


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, image):
        h, w, _ = image.shape
        if h == self.new_h and w == self.new_w:
            return image
        image = imresize(image, (self.new_h, self.new_w))
        return image