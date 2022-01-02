import torch
import torch.nn as nn
import torch.nn.functional as F


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)


class PseSupMaskLoss(nn.Module):
    def __init__(self):
        super(PseSupMaskLoss, self).__init__()
        self.THRES_EPI = 0.3

    def forward(self, depth_est, pse_depth_gt, epi, filter_mask):
        # pse_depth_gt: B x 216 x 288
        # epi: B x 128 x 160
        height, width = pse_depth_gt.shape[1], pse_depth_gt.shape[2]

        confidence = torch.exp(-epi)  # 归一化为置信度，范围为0-1
        confidence = confidence.unsqueeze(dim=1)  # B x 1 x 128 x 160
        confidence = F.interpolate(confidence, size=(height, width))  # B x 1 x 216 x 288
        confidence = confidence.squeeze(dim=1)  # B x 216 x 288
        confidence_mask = confidence > self.THRES_EPI
        confidence_mask = confidence_mask.float()

        depth_est = depth_est.unsqueeze(dim=1)  # B x 1 x 128 x 160
        depth_est = F.interpolate(depth_est, size=(height, width))  # B x 1 x 216 x 288
        depth_est = depth_est.squeeze(dim=1)  # B x 216 x 288

        filter_mask = filter_mask.unsqueeze(dim=1)  # B x 1 x 128 x 160
        filter_mask = F.interpolate(filter_mask, size=(height, width))  # B x 1 x 216 x 288
        filter_mask = filter_mask.squeeze(dim=1)  # B x 216 x 288

        mask = filter_mask * confidence_mask

        loss = mvsnet_loss(depth_est * confidence, pse_depth_gt * confidence, mask)
        return loss


class PseSupLoss(nn.Module):
    def __init__(self):
        super(PseSupLoss, self).__init__()
        self.THRES_EPI = 0.3

    def forward(self, depth_est, pse_depth_gt, epi):
        # pse_depth_gt: B x 216 x 288
        # epi: B x 128 x 160
        height, width = pse_depth_gt.shape[1], pse_depth_gt.shape[2]

        confidence = torch.exp(-epi)  # 归一化为置信度，范围为0-1
        confidence = confidence.unsqueeze(dim=1)  # B x 1 x 128 x 160
        confidence = F.interpolate(confidence, size=(height, width))  # B x 1 x 216 x 288
        confidence = confidence.squeeze(dim=1)  # B x 216 x 288
        confidence_mask = confidence > self.THRES_EPI
        confidence_mask = confidence_mask.float()

        depth_est = depth_est.unsqueeze(dim=1)  # B x 1 x 128 x 160
        depth_est = F.interpolate(depth_est, size=(height, width))  # B x 1 x 216 x 288
        depth_est = depth_est.squeeze(dim=1)  # B x 216 x 288

        filter_mask = torch.ones_like(depth_est)  # B x 216 x 288

        mask = filter_mask * confidence_mask

        loss = mvsnet_loss(depth_est * confidence, pse_depth_gt * confidence, mask)
        return loss