# -*- coding: utf-8 -*-
# @Time    : 2020/09/08 17:06
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : ap_transforms
# @Software: PyCharm

import numpy as np
import torch
from torchvision import transforms
from PIL import ImageFilter


def get_ap_transforms(cfg):
    t = [ToPILImage()]
    if cfg.cj:
        t.append(ColorJitter(brightness=cfg.cj_bri, contrast=cfg.cj_con,
                                      saturation=cfg.cj_sat, hue=cfg.cj_hue))
    if cfg.gblur:
        t.append(RandomGaussianBlur(0.5, 3))
    t.append(ToTensor())
    if cfg.gamma:
        t.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))
    return transforms.Compose(t)


# from https://github.com/visinf/irr/blob/master/datasets/transforms.py
class ToPILImage(transforms.ToPILImage):
    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im) for im in imgs]


class ColorJitter(transforms.ColorJitter):
    def __call__(self, imgs):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return [transform(im) for im in imgs]


class ToTensor(transforms.ToTensor):
    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im) for im in imgs]


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, imgs):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return [self.adjust_gamma(im, gamma, self._clip_image) for im in imgs]


class RandomGaussianBlur():
    def __init__(self, p, max_k_sz):
        self.p = p
        self.max_k_sz = max_k_sz

    def __call__(self, imgs):
        if np.random.random() < self.p:
            radius = np.random.uniform(0, self.max_k_sz)
            imgs = [im.filter(ImageFilter.GaussianBlur(radius)) for im in imgs]
        return imgs