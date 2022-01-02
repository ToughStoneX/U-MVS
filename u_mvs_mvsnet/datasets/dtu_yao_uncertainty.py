from torch.utils.data import Dataset
import numpy as np
import os
import math
from PIL import Image
import cv2
from torchvision import transforms
import torch

from datasets.data_io import *


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

    def __call__(self, img):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)


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

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

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

    def __len__(self):
        return len(self.metas)

    def center_image(self, img):
        """ normalize image input """
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def mask_depth_image(self, depth_image, min_depth, max_depth):
        """ mask out-of-range pixel to zero """
        ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
        ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
        depth_image = np.expand_dims(depth_image, 2)
        return depth_image

    def load_cam(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 256
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        return cam

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

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

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        ###### read input data ######
        images = []
        images_aug = []
        images_aug2 = []
        images_seg = []
        cams = []
        proj_matrices = []
        intrinsics_list = []
        extrinsics_list = []
        epi_list = []
        dep_unc_list = []
        # for view in range(self.nviews):
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

            image = self.center_image(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB))
            image_aug = self.read_img_aug(img_filename)
            image_aug2 = self.read_img_aug(img_filename)
            image_seg = self.read_img_seg(img_filename)
            cam = self.load_cam(open(proj_mat_filename))
            cam[1][3][1] = cam[1][3][1] * self.interval_scale
            images.append(image)
            images_seg.append(image_seg)
            images_aug.append(image_aug)
            images_aug2.append(image_aug2)
            cams.append(cam)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                depth_image = self.read_depth(depth_filename)
                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (self.ndepths - 2) * (cams[0][1, 3, 1])
                depth_image =self.mask_depth_image(depth_image, depth_start, depth_end)
                mask = self.read_img(mask_filename)

                epi_filename = os.path.join(os.getcwd(), 'uncertainty', scan, 'epistemic', '{:0>8}.npz'.format(vid))
                epi = np.load(epi_filename)['arr_0']  # 128 x 160
                dep_unc_filename = os.path.join(os.getcwd(), 'uncertainty', scan, 'depth', '{:0>8}.npz'.format(vid))
                dep_unc = np.load(dep_unc_filename)['arr_0'] # 216 x 288

        images = np.stack(images).transpose([0, 3, 1, 2])
        images_aug = np.stack(images_aug)
        images_aug2 = np.stack(images_aug2)
        images_seg = np.stack(images_seg)
        cams = np.stack(cams)
        proj_matrices = np.stack(proj_matrices)
        intrinsics_list = np.stack(intrinsics_list)
        extrinsics_list = np.stack(extrinsics_list)

        sample = {"imgs": images,
                "imgs_aug": images_aug,
                "imgs_aug2": images_aug2,
                "imgs_seg": images_seg,
                "proj_matrices": proj_matrices,
                "intrinsics": intrinsics_list,
                "extrinsics": extrinsics_list,
                "mask": mask,
                "cams": cams,
                "depth": depth_image,
                "depth_values": depth_values,
                "depth_start": depth_start,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}

        if self.mode == 'train':
            sample['epi'] = epi
            sample['dep_unc'] = dep_unc
        return sample


if __name__ == "__main__":
    datapath = "D:\\BaiduNetdiskDownload\\mvsnet\\training_data\\dtu_training"
    listfile = "E:\\PycharmProjects\\un_mvsnet_pytorch\\lists\\dtu\\train.txt"
    train_dataset = MVSDataset(datapath, listfile, "train", nviews=3, ndepths=192, interval_scale=1.06)
    print('dataset length: {}'.format(len(train_dataset)))
    item = train_dataset[1000]
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("cams", item["cams"].shape)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(item["depth"][:, :, 0], cmap='gray')
    plt.tight_layout()
    plt.show()