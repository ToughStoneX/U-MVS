# -*- coding: utf-8 -*-
# @Time    : 2020/09/08 20:40
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : get_model
# @Software: PyCharm

from models.pwclite import PWCLite


def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWCLite(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model