# -*- coding: utf-8 -*-
# @Time    : 2020/09/25 20:32
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : gen_flow
# @Software: PyCharm

import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import torch
from easydict import EasyDict
from torchvision import transforms
import flow_vis
import os
from pathlib import Path

from transforms import sep_transforms
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
    # parser.add_argument('-i', '--img_list', nargs='+',
    #                     default=['examples/img13.png', 'examples/img14.png'])
    # parser.add_argument('-c', '--cam_list', nargs='+',
    #                     default=['examples/cam9.png', 'examples/cam10.png'])
    parser.add_argument('-p', '--datapath', type=str, default='')
    parser.add_argument('--listfile', type=str, default='../lists/dtu/train.txt')
    parser.add_argument('--output_dir', type=str, default='flow')
    parser.add_argument('--save_jpg', action='store_true', help='save flow to images')
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }
    ts = TestHelper(cfg)

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True, parents=True)

    # process dataset
    with open(args.listfile) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    # scans
    metas = []
    for scan in scans:
        pair_file = "Cameras/pair.txt"
        # read the pair file
        with open(os.path.join(args.datapath, pair_file)) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                # light conditions 3
                metas.append((scan, 3, ref_view, src_views))
    print("metas:", len(metas))

    for i, meta in enumerate(metas):
        print('[{}/{}]'.format(i, len(metas)))
        scan, light_idx, ref_view, src_views = meta
        scan_dir = out_dir.joinpath(scan)
        if not scan_dir.exists():
            scan_dir.mkdir(exist_ok=True, parents=True)

        view_ids = [ref_view] + src_views[:6]
        imgs = []
        for j, vid in enumerate(view_ids):
            img_filename = os.path.join(args.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            img = imageio.imread(img_filename).astype(np.float32)
            imgs.append(img)

        for v1 in range(len(imgs)):
            for v2 in range(v1+1, len(imgs)):
                vid1, vid2 = view_ids[v1]+1, view_ids[v2]+1
                print('{}-{}'.format(vid1, vid2))

                flows_12_path = scan_dir.joinpath('flow_{:02d}_{:02d}.npz'.format(vid1, vid2))
                flows_21_path = scan_dir.joinpath('flow_{:02d}_{:02d}.npz'.format(vid2, vid1))
                if flows_12_path.exists() and flows_21_path.exists():
                    continue

                img1, img2 = imgs[v1], imgs[v2]
                res_dict = ts.run([img1, img2])
                flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
                flow_12, flow_21 = flows_12[0], flows_21[0]
                # print('flow_12: {} flow_21: {}'.format(flow_12.shape, flow_21.shape))

                if not flows_12_path.exists():
                    np_flow_12 = flow_12.detach().cpu().numpy()
                    np.savez_compressed(str(flows_12_path), np_flow_12)
                if not flows_21_path.exists():
                    np_flow_21 = flow_21.detach().cpu().numpy()
                    np.savez_compressed(str(flows_21_path), np_flow_21)

                if args.save_jpg:
                    np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
                    np_flow_21 = flow_21[0].detach().cpu().numpy().transpose([1, 2, 0])
                    vis_flow_12 = flow_vis.flow_to_color(np_flow_12, convert_to_bgr=False)
                    vis_flow_21 = flow_vis.flow_to_color(np_flow_21, convert_to_bgr=False)
                    vis_flow_12_path = scan_dir.joinpath('flow_{:02d}_{:2d}.jpg'.format(vid1, vid2))
                    vis_flow_21_path = scan_dir.joinpath('flow_{:02d}_{:2d}.jpg'.format(vid2, vid1))
                    misc.imsave(vis_flow_12_path, vis_flow_12)
                    misc.imsave(vis_flow_21_path, vis_flow_21)

