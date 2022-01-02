import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import args
from losses.modules import *
from losses.homography import *


def depth_pose2flow(depth, ref_in, ref_ex, src_in, src_ex):
    """
    :param depth:    B x H x W
    :param ref_in:   B x 3 x 3
    :param ref_ex:   B x 4 x 4
    :param src_in:   B x 3 x 3
    :param src_ex:   B x 4 x 4
    :return:  B x 2 x H x W
    """
    batch = depth.shape[0]
    height, width = depth.shape[1], depth.shape[2]

    src_proj = torch.matmul(src_in, src_ex[:, 0:3, :])  # B x 3 x 4
    ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])  # B x 3 x 4
    last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(len(src_in), 1, 1).cuda()
    src_proj = torch.cat((src_proj, last), 1)  # B x 4 x 4
    ref_proj = torch.cat((ref_proj, last), 1)  # B x 4 x 4
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    grid = torch.stack((x, y))  # [2, H*W]
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    d = depth.reshape(batch, height * width).unsqueeze(1)  # [B, 1, H*W]
    rot_depth_xyz = rot_xyz * d  # [B, 3, H*W]
    proj_xyz = rot_depth_xyz + trans  # [B, 3, H*W]
    proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :].clamp(min=1e-3)  # [B, 2, H*W]
    flow = proj_xy - grid.unsqueeze(0)  # [B, 2, H*W]
    return flow.reshape(batch, 2, height, width)


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


class UnFlowLoss(nn.Module):
    def __init__(self):
        super(UnFlowLoss, self).__init__()

    # def forward(self, imgs, cams, depth):
    def forward(self, flow12, flow21, depth, ref_in, ref_ex, src_in, src_ex, mask):
        # flow12: [B, V-1, 2, 512, 640]
        # flow21: [B, V-1, 2, 512, 640]
        loss = []
        flow_list = []
        mask_list = []
        flows_viz = []

        depth_mask = mask.unsqueeze(dim=1)
        depth_mask = F.interpolate(depth_mask, scale_factor=(4, 4), mode='bilinear')
        # convert depth and pose to flow
        for src_idx in range(args.view_num - 1):
            # 取出计算好的对应视角的光流
            flow12_src = flow12[:, src_idx]
            flow21_src = flow21[:, src_idx]

            # 将深度图转换成对应视角之间的光流
            flow = depth_pose2flow(depth, ref_in, ref_ex, src_in[:, src_idx], src_ex[:, src_idx])
            h, w = flow12_src.shape[2], flow12_src.shape[3]
            flow = resize_flow(flow, (h, w))

            # 根据光流计算遮挡掩码
            occu_mask1 = 1 - get_occu_mask_bidirection(flow12_src, flow21_src)
            occu_mask1 = occu_mask1.detach()

            # 光流误差
            flow_diff = (flow - flow12_src).abs() * occu_mask1 + 1e4 * (1 - occu_mask1)
            flow_list.append(torch.unsqueeze(flow_diff, dim=0))

            flows_viz.append(flow.detach().cpu().numpy())

        # 从所有视角取误差最小的那个
        flow_list = torch.cat(flow_list, dim=0)
        flow_min = flow_list.min(dim=0)[0]
        mask = flow_min < 1e4 - 1
        mask = mask.float()
        mask = mask * depth_mask
        flow_min = flow_min * mask
        loss = flow_min.mean() / mask.mean()

        return loss, flows_viz

