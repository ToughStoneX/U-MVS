# -*- coding: utf-8 -*-
# @Time    : 2020/09/25 19:27
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : inference
# @Software: PyCharm

import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms
import flow_vis

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair, with_bk=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img13.png', 'examples/img14.png'])
    parser.add_argument('-c', '--cam_list', nargs='+',
                        default=['examples/cam9.png', 'examples/cam10.png'])
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    ts = TestHelper(cfg)

    imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
    h, w = imgs[0].shape[:2]

    res_dict = ts.run(imgs)
    flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
    flow_12, flow_21 = flows_12[0], flows_21[0]
    print('flow_12: {} flow_21: {}'.format(flow_12.shape, flow_21.shape))

    occu_mask1 = 1 - get_occu_mask_bidirection(flow_12, flow_21)
    occu_mask2 = 1 - get_occu_mask_bidirection(flow_21, flow_12)
    # occu_mask1 = 1 - get_occu_mask_backward(flow_21, th=0.4)
    # occu_mask2 = 1 - get_occu_mask_backward(flow_12, th=0.4)
    flow_12_occu = flow_12 * occu_mask1.repeat(1, 2, 1, 1)
    flow_21_occu = flow_21 * occu_mask2.repeat(1, 2, 1, 1)
    occu_mask1, occu_mask2 = occu_mask1.float(), occu_mask2.float()
    print('occu_mask1: {} {}-{}'.format(occu_mask1.shape, occu_mask1.min(), occu_mask1.max()))
    print('occu_mask2: {} {}-{}'.format(occu_mask2.shape, occu_mask2.min(), occu_mask2.max()))
    occu_mask1, occu_mask2 = occu_mask1[0, 0].detach().cpu().numpy(), occu_mask2[0, 0].detach().cpu().numpy()

    flow_12 = resize_flow(flow_12, (h, w))
    np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
    flow_21 = resize_flow(flow_21, (h, w))
    np_flow_21 = flow_21[0].detach().cpu().numpy().transpose([1, 2, 0])
    flow_12_occu = resize_flow(flow_12_occu, (h, w))
    np_flow_12_occu = flow_12_occu[0].detach().cpu().numpy().transpose([1, 2, 0])
    flow_21_occu = resize_flow(flow_21_occu, (h, w))
    np_flow_21_occu = flow_21_occu[0].detach().cpu().numpy().transpose([1, 2, 0])

    vis_flow_12 = flow_vis.flow_to_color(np_flow_12, convert_to_bgr=False)
    vis_flow_21 = flow_vis.flow_to_color(np_flow_21, convert_to_bgr=False)
    vis_flow_12_occu = flow_vis.flow_to_color(np_flow_12_occu, convert_to_bgr=False)
    vis_flow_21_occu = flow_vis.flow_to_color(np_flow_21_occu, convert_to_bgr=False)
    # vis_flow_12 = flow_to_image(np_flow_12)
    # vis_flow_21 = flow_to_image(np_flow_21)
    # vis_flow_12_occu = flow_to_image(np_flow_12_occu)
    # vis_flow_21_occu = flow_to_image(np_flow_21_occu)
    # print('vis_flow: {} {}-{}'.format(vis_flow.shape, vis_flow.min(), vis_flow.max()))

    misc.imsave('vis_flow_12.jpg', vis_flow_12)
    misc.imsave('vis_flow_21.jpg', vis_flow_21)
    misc.imsave('occu_mask1.jpg', occu_mask1)
    misc.imsave('occu_mask2.jpg', occu_mask2)
    misc.imsave('vis_flow_12_occu.jpg', vis_flow_12_occu)
    misc.imsave('vis_flow_21_occu.jpg', vis_flow_21_occu)


    # fig = plt.figure()
    # plt.imshow(vis_flow)
    # plt.show



