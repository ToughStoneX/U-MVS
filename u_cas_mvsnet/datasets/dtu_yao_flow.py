from torch.utils.data import Dataset
import numpy as np
import torch
import os, cv2, time, math
from PIL import Image
from torchvision import transforms

from datasets.data_io import *
from datasets.utils import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.define_transforms()


    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas


    def define_transforms(self):
        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_seg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.metas)


    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval


    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    def read_img_seg(self, filename):
        img = Image.open(filename)
        return self.transform_seg(img)


    def read_img_aug(self, filename):
        img = Image.open(filename)
        img = self.transform_aug(img)
        return img


    def center_image(self, img):
        """ normalize image input """
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)


    def prepare_img(self, hr_img):
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        # downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        # crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop


    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms


    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def read_depth_hr(self, filename):
        # read pfm depth file
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms


    def load_flow(self, scan, v1, v2):
        forward_flow_name = 'flow_{:02d}_{:02d}.npz'.format(v1 + 1, v2 + 1)
        backward_flow_name = 'flow_{:02d}_{:02d}.npz'.format(v2 + 1, v1 + 1)
        forward_flow_path = os.path.join(os.getcwd(), 'arflow', 'flow', scan, forward_flow_name)
        backward_flow_path = os.path.join(os.getcwd(), 'arflow', 'flow', scan, backward_flow_name)
        forward_flow = np.load(forward_flow_path)['arr_0']  # [1, 2, 512, 640]
        backward_flow = np.load(backward_flow_path)['arr_0']  # [1, 2, 512, 640]
        forward_flow, backward_flow = forward_flow[0], backward_flow[0]
        return forward_flow, backward_flow


    def load_flow_h(self, scan, v1, v2):
        forward_flow_name = 'flow_{:02d}_{:02d}.npz'.format(v1 + 1, v2 + 1)
        backward_flow_name = 'flow_{:02d}_{:02d}.npz'.format(v2 + 1, v1 + 1)
        forward_flow_path = os.path.join(os.getcwd(), 'arflow', 'flow2', scan, forward_flow_name)
        backward_flow_path = os.path.join(os.getcwd(), 'arflow', 'flow2', scan, backward_flow_name)
        forward_flow = np.load(forward_flow_path)['arr_0']  # [1, 2, 512, 640]
        backward_flow = np.load(backward_flow_path)['arr_0']  # [1, 2, 512, 640]
        forward_flow, backward_flow = forward_flow[0], backward_flow[0]
        return forward_flow, backward_flow


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        center_imgs = []
        imgs_aug = []
        mask = None
        depth_values = None
        proj_matrices = []
        forward_flow_list = []
        backward_flow_list = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            # img = self.read_img(img_filename)
            image_seg = self.read_img_seg(img_filename)
            center_img = self.center_image(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB))
            aug_img = self.read_img_aug(img_filename)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)

                # get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms
            else:  # source view
                forward_flow, backward_flow = self.load_flow(scan, ref_view, vid)
                # forward_flow, backward_flow = self.load_flow_h(scan, ref_view, vid)
                forward_flow_list.append(forward_flow)
                backward_flow_list.append(backward_flow)

            # imgs.append(img)
            imgs.append(image_seg)
            center_imgs.append(center_img)
            imgs_aug.append(aug_img)

        # all
        # imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        imgs = np.stack(imgs)
        center_imgs = np.stack(center_imgs).transpose([0, 3, 1, 2])
        imgs_aug = torch.stack(imgs_aug)
        forward_flow_list = np.stack(forward_flow_list)  # [V-1, 2, 512, 640]
        backward_flow_list = np.stack(backward_flow_list)  # [V-1, 2, 512, 640]
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()  # [B, V, 2, 4, 4]]
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "center_imgs": center_imgs,
                "imgs_aug": imgs_aug,
                "forward_flow": forward_flow_list,
                "backward_flow": backward_flow_list}