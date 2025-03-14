# Modified from the MONAI SwinUNETR/BTCV code: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV

import argparse
import os
from functools import partial
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import torch.nn as nn
import socket

parser = argparse.ArgumentParser(description="Prostate triage UNet segmentation training pipeline")
parser.add_argument("--checkpoint", action="store_true", help="resume training from previous checkpoint")
parser.add_argument("--checkpoint_epoch", default=-1, type=int, help="epoch of the checkpoint to load")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--append_logdir", default="", type=str, help="append to logdir")
parser.add_argument("--data_dir", default="/home/user/Documents/coreg", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=27, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=5, type=int, help="number of workers")
parser.add_argument("--valworkers", default=1, type=int, help="number of val workers")
parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch factor for dataloader')

parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=0.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=0.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=256, type=float, help="roi in x direction")
parser.add_argument("--roi_y", default=256, type=float, help="roi in y direction")
parser.add_argument("--roi_z", default=32, type=float, help="roi in z direction")
parser.add_argument('--min_percentile', default=0.0, type=float, help='min percentile for intensity clipping')
parser.add_argument('--max_percentile', default=98.0, type=float, help='max percentile for intensity clipping')

parser.add_argument('--randflip_prob', default=0.5, type=float, help='probability of random flip')
parser.add_argument('--randrotate_prob', default=0.5, type=float, help='probability of random rotate')
parser.add_argument('--randaffine_prob', default=0.5, type=float, help='probability of random affine')
parser.add_argument('--randzoom_prob', default=0.5, type=float, help='probability of random zoom')
parser.add_argument('--randnoise_prob', default=0.2, type=float, help='probability of random noise')
parser.add_argument('--randscale_prob', default=0.2, type=float, help='probability of random scale')
parser.add_argument('--randshift_prob', default=0.2, type=float, help='probability of random shift')

parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument('--startfeatures', default=32, type=int, help='number of UNet starting features')
parser.add_argument('--alpha', default=0.75, type=float, help='alpha for Focal Loss or CE')

parser.add_argument('--nocache', action='store_true', help='do not use cache')
parser.add_argument('--cache_dir', default='/home/user/Documents/cache', type=str, help='cache directory')

parser.add_argument('--max_grad_norm', default=3.0, type=float, help='max gradient norm for gradient clipping')
        

def main():
    args = parser.parse_args()
    args.test = False
    if torch.cuda.device_count() > 1:
        args.distributed = True
        print("Using distributed with total gpus", torch.cuda.device_count())
    else:
        args.distributed = False

    args.noamp = True
    
    args.amp = not args.noamp # Defaults to True
    if args.amp:
        print("Using AMP")
    args.noearlystop = True
    args.use_checkpoint = True
    
    args.save_checkpoint = True # Always save checkpoint when available

    args.save_model_every = args.val_every

    if not (args.roi_x == 256 and args.roi_y == 256 and args.roi_z == 32):
        raise ValueError("Only 256x256x32 supported")
    
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu_i=0, args=args)


def main_worker(gpu_i, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu_i
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu_i
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    else:
        args.rank = 0
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    channels = (args.startfeatures, args.startfeatures*2, args.startfeatures*4, args.startfeatures*8, args.startfeatures*16)
    model = UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=channels,
            strides=(2,2,2,2),
            num_res_units=2,
            dropout=0.1,
        )
    model.cuda(args.gpu)

    if args.checkpoint: # Load checkpoint
        if args.checkpoint_epoch != -1:
            checkpoint_dir = os.path.join(args.logdir, str(args.checkpoint_epoch))
            model_name = "model_" + str(args.checkpoint_epoch) + ".pt"
            checkpoint_path = os.path.join(checkpoint_dir, model_name)
            checkpoint = torch.load(checkpoint_path)
        else: # Attempt to load the "model_last.pt" at args.logdir
            checkpoint_path = os.path.join(args.logdir, "model_last.pt")
            checkpoint = torch.load(checkpoint_path)
        model_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]
        start_epoch = checkpoint["epoch"]+1
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(model_dict)
        
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(checkpoint_path, epoch, best_acc))
    else:
        best_acc = 0
        start_epoch = 0

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    # Optimizer
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, amsgrad=True)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, amsgrad=True)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    
    if args.checkpoint: # Load optimizer
        if "optimizer" in checkpoint and not args.notscheduler:
            optimizer.load_state_dict(checkpoint["optimizer"])

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs, last_epoch=-1, 
        )
        if args.checkpoint and "scheduler" in checkpoint and not args.notscheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = epoch
            scheduler.step(epoch=start_epoch)
        elif args.checkpoint and "scheduler" in checkpoint and args.notscheduler:
            scheduler.last_epoch = epoch
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, last_epoch=-1, )
        if args.checkpoint:
            if "scheduler" in checkpoint and not args.notscheduler:
                scheduler.load_state_dict(checkpoint["scheduler"])
                scheduler.last_epoch = epoch
            elif "scheduler" in checkpoint and args.notscheduler:
                scheduler.last_epoch = epoch
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "exponential":
        lr_gamma = 0.99
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        scheduler = None

    args.logdir = args.logdir + args.append_logdir
    reduction = 'sum'

    if args.out_channels == 2:
        dice_loss = FocalLoss(to_onehot_y=True, use_softmax=True, alpha=args.alpha, gamma=2.0, reduction=reduction)
        post_label = AsDiscrete(to_onehot=args.out_channels)
        post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, ignore_empty=False)
    else:
        dice_loss = FocalLoss(use_softmax=False, alpha=args.alpha, gamma=2.0, reduction=reduction)
        post_label = AsDiscrete(threshold=0.5)
        post_pred = AsDiscrete(threshold=0.5)
        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, ignore_empty=False)
    
    if (args.roi_x==256 and args.roi_y==256 and args.roi_z==32):
        model_inferer = None
    else:
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
            mode='gaussian',
        )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model.to(device=torch.device("cuda",args.gpu), non_blocking=True)

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
