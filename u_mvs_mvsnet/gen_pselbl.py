import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
import gc
import sys
import datetime
import random
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
import imageio

from models.mvsnet_uncertainty import MVSNet, mvsnet_loss
from losses.unsup_loss_uncertainty import *
from utils import *
from config import args, device

cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

if args.mode == "test":
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(os.path.join(args.logdir, 'test'))

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=True)

# model, optimizer
model = MVSNet(refine=args.refine).to(device)
model = nn.DataParallel(model)
criterion = UnSupLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

inv_seg_transform = transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TrainImgLoader):
        start_time = time.time()
        estimate_uncertainty(sample, detailed_summary=True)
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TrainImgLoader), time.time() - start_time))
    print("done.", avg_test_scalars)


def estimate_uncertainty(sample, detailed_summary=True):
    depth_samples = []
    model.train()
    sample_cuda = tocuda(sample)

    # propagate T(args.test_trials=20) times forward pass to estimate the epistemic uncertainty
    for i in range(args.test_trials):
        optimizer.zero_grad()
        outputs = model(sample_cuda["imgs_seg"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        depth_est = outputs["depth"]
        log_var = outputs["log_var"]
        standard_loss = criterion(sample_cuda["imgs"], sample_cuda["cams"], depth_est, log_var)
        loss = standard_loss
        loss.backward()
        depth_samples.append(depth_est.unsqueeze(dim=0))
    depth_samples = torch.cat(depth_samples, dim=0)
    depth_mean = torch.mean(depth_samples, dim=0)
    depth_var = torch.std(depth_samples, dim=0)  # epistemic uncertainty

    # 1 forward pass with no dropout for aleatoric_uncertainty
    model.eval()
    outputs = model(sample_cuda["imgs_seg"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est2 = outputs["depth"]
    log_var2 = outputs["log_var"]  # the original output is logarithmic form of aleatoric uncertainty
    # normalize variances
    var2 = torch.exp(log_var2)  # aleatoric uncertainty

    print('aleatoric uncertainty: {} - {}'.format(var2.min(), var2.max()))
    print('epistemic uncertainty: {} - {}'.format(depth_var.min(), depth_var.max()))

    save_uncertainty_maps(depth_var, var2, sample["filename"])
    save_depth_map(depth_mean, sample["filename"])


def save_uncertainty_maps(epi_uncertain, ale_uncertain, filenames):
    for filename, epi_unc, ale_unc in zip(filenames, epi_uncertain, ale_uncertain):
        epi_filename = os.path.join(args.outdir, filename.format('epistemic', '.npz'))
        ale_filename = os.path.join(args.outdir, filename.format('aleatoric', '.npz'))
        epi = epi_unc.detach().cpu().numpy()  # [128, 160]
        ale = ale_unc.detach().cpu().numpy()  # [128, 160]
        if os.path.exists(epi_filename):
            pass
        else:
            os.makedirs(epi_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(ale_filename.rsplit('/', 1)[0], exist_ok=True)
            np.savez_compressed(epi_filename, epi)
            np.savez_compressed(ale_filename, ale)


def save_depth_map(depth, filenames):
    for filename, dep in zip(filenames, depth):
        depth_filename = os.path.join(args.outdir, filename.format('depth', '.npz'))
        depth_mean = dep.detach().cpu().numpy()
        if os.path.exists(depth_filename):
            pass
        else:
            os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            np.savez_compressed(depth_filename, depth_mean)


if __name__ == '__main__':
    if args.mode == "test":
        test()