# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import socket
import sys
from datetime import datetime
import time
import math
import json
import numpy as np
import yaml
from carp_loss import PatchCrossEntropy, CrossEntropy
from koleo_loss import KoLeoLoss
from memory_bank import MemoryBank
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from models.head import iBOTHead
from loader import ImageFolderMask
# from evaluation.unsupervised.unsup_cls import eval_pred


def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_large'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--class_memory_size', default=65536, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
                        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block',
                        type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="""We use gradient 
                        accumulation to simulate large batch sizes in small gpus.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=100, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument("--local_rank", type=int,
    #                     help="Please ignore and do not set this argument.")
    parser.add_argument("--num_class_experts", default=9, type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument("--num_patch_experts", default=1, type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument("--smoothing", default=0.1, type=float,
                        help="Label smoothing hyper-param.")
    parser.add_argument('--patch_partition_size',
                        type=int, default=512, help="...")
    parser.add_argument('--patch_memory_size', type=int,
                        default=8192, help="...")
    parser.add_argument('--num_class_tasks', type=int, default=4, help="...")
    parser.add_argument('--num_patch_tasks', type=int, default=1, help="...")
    parser.add_argument("--class_partition_size", default=64,
                        type=int, help="The size of the subgroups.")
    parser.add_argument("--koleo_loss_weight", default=0.0,
                        type=float, help="Weight for the koleo loss contribution")
    parser.add_argument("--bottleneck_dim", default=256,
                        type=int, help="Dimensionality of the embedding vector.")
    parser.add_argument("--student_temp", default=0.1, type=float,
                        help="Temperature for student logits prior to softmax.")
    parser.add_argument('--print_freq', default=50, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--resume_from_dir', default=".",
                        type=str, help='Path to save logs and checkpoints.')
    return parser


def train_ibot(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    args.total_crops_number = args.local_crops_number + args.global_crops_number

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset = ImageFolderMask(
        args.data_path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        # persistent_workers=True,
        # prefetch_factor=2
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        in_dim=embed_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=args.bottleneck_dim,
        mlp_bias=True,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            in_dim=embed_dim,
            bottleneck_dim=args.bottleneck_dim,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
        'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(
        student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing memories ... ============
    class_memory_bank = MemoryBank(K=args.class_memory_size, num_experts_per_concept=args.num_class_experts,
                                   partition_size=args.class_partition_size, out_dim=args.bottleneck_dim,
                                   smoothing=args.smoothing, num_tasks=args.num_class_tasks).cuda()

    patch_memory_bank = MemoryBank(K=args.patch_memory_size, num_experts_per_concept=args.num_patch_experts,
                                   partition_size=args.patch_partition_size, out_dim=args.bottleneck_dim,
                                   smoothing=args.smoothing, num_tasks=args.num_patch_tasks).cuda()

    print(class_memory_bank)
    print(patch_memory_bank)

    summary_writer = None
    if utils.is_main_process():
        summary_writer = SummaryWriter()

        shutil.copyfile("main_ibot.py", os.path.join(
            summary_writer.log_dir, "main_ibot.py")
        )
        shutil.copyfile("utils.py", os.path.join(
            summary_writer.log_dir, "utils.py")
        )
        # shutil.copyfile(
        #     "carp_loss.py", os.path.join(
        #         summary_writer.log_dir, "carp_loss.py")
        # )
        shutil.copyfile("memory_bank.py", os.path.join(
            summary_writer.log_dir, "memory_bank.py")
        )
        shutil.copyfile("./models/vision_transformer.py", os.path.join(
            summary_writer.log_dir, "vision_transformer.py")
        )
        # shutil.copyfile(
        #     "koleo_loss.py", os.path.join(
        #         summary_writer.log_dir, "koleo_loss.py")
        # )
        stats_file = open(
            os.path.join(summary_writer.log_dir, "stats.txt"), "a", buffering=1
        )
        print(" ".join(sys.argv), flush=True)
        print(" ".join(sys.argv), file=stats_file, flush=True)
        with open(os.path.join(summary_writer.log_dir, "metadata.txt"), "a") as f:
            yaml.dump(args, f, allow_unicode=True)
            f.write(str(student))
            f.write(str(teacher))

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params_groups, fused=True)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        # to use with convnet and large batches
        optimizer = utils.LARS(params_groups)
    optimizer.zero_grad()

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()
                   ) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    teacher_temp_schedule = utils.cosine_scheduler(
        base_value=args.teacher_temp,
        final_value=args.teacher_temp,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=args.warmup_teacher_temp_epochs,
        start_warmup_value=args.warmup_teacher_temp
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))

    print(f"Loss, optimizer and schedulers ready.")
    koleo_loss = KoLeoLoss().cuda()
    cross_entropy = CrossEntropy().cuda()
    patch_cross_entropy = PatchCrossEntropy().cuda()

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}

    utils.restart_from_checkpoint(
        os.path.join(args.resume_from_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        class_memory_bank=class_memory_bank,
        patch_memory_bank=patch_memory_bank,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler)

    start_epoch = to_restore["epoch"]

    # torch.compile modules
    student_opt = torch.compile(student)
    teacher_opt = torch.compile(teacher)
    class_memory_bank_opt = torch.compile(class_memory_bank)
    patch_memory_bank_opt = torch.compile(patch_memory_bank)

    start_time = time.time()
    print("Starting SOVE training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_one_epoch(student_opt, teacher_opt, teacher_without_ddp, cross_entropy, patch_cross_entropy,
                        data_loader, optimizer, lr_schedule, wd_schedule, teacher_temp_schedule, momentum_schedule,
                        epoch, fp16_scaler, koleo_loss, class_memory_bank_opt, patch_memory_bank_opt, summary_writer, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_memory_bank': class_memory_bank.state_dict(),
            'patch_memory_bank': patch_memory_bank.state_dict(),
            'epoch': epoch + 1,
            'args': args
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        if utils.is_main_process():
            utils.save_on_master(save_dict, os.path.join(
                summary_writer.log_dir, 'checkpoint.pth'))
            if args.saveckp_freq and ((epoch + 1) % args.saveckp_freq == 0) and epoch:
                utils.save_on_master(save_dict, os.path.join(
                    summary_writer.log_dir, f'checkpoint{epoch:04}.pth'))

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, cross_entropy, patch_cross_entropy, data_loader,
                    optimizer, lr_schedule, wd_schedule, teacher_temp_schedule, momentum_schedule, epoch,
                    fp16_scaler, koleo_loss, class_memory_bank, patch_memory_bank, summary_writer, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    cls_loss = AverageMeter('CLS Loss', ':.4e')
    patch_loss = AverageMeter('Patch Loss', ':.4e')
    # ko_collisions = AverageMeter('KO Collisions', ':.4e')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, learning_rates, losses, cls_loss, patch_loss],
        prefix="Epoch: [{}]".format(epoch))

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(
        names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(
        names_k, params_k) if name_k in names_common]

    end = time.time()
    for i, (images, labels, masks) in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule
        data_time.update(time.time() - end)

        it = len(data_loader) * epoch + i  # global training iteration

        lr = lr_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        learning_rates.update(lr)

        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]
        labels = labels.cuda(non_blocking=True)

        # The 'identity_mask' mask is used disable 'masked_im_modeling'.
        # Disabling 'masked_im_modeling' using :
        #   student.module.backbone.masked_im_modeling = False
        # as done in the original implementation, raises problems with torch.compile()
        # because the 'self.masked_embed' variable in the
        # vision transformer class will not receive gradients.
        identity_mask = [torch.zeros_like(
            mask, dtype=torch.bool) for mask in masks[args.global_crops_number:]]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # get global views
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(
                images[:args.global_crops_number], mask=masks[:args.global_crops_number])

            # get local views
            # student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:], mask=identity_mask)[
                0] if len(images) > args.global_crops_number else None
            # student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            student_cls, student_patch = student_output
            teacher_cls, teacher_patch = teacher_output

            # koleo loss must come before random partition
            ko = 0
            for p in student_cls.chunk(2):
                ko += koleo_loss(p)
            ko /= 2

            if student_local_cls is not None:
                student_cls = torch.cat([student_cls, student_local_cls])

            class_memory_output = class_memory_bank(student_cls,
                                                    teacher_cls,
                                                    student_temp=args.student_temp,
                                                    teacher_temp=teacher_temp)

            student_prt_probs_list, teacher_prt_probs_list = class_memory_output

            assert len(student_prt_probs_list) == len(
                teacher_prt_probs_list)

            cls_ce = 0
            for student_prt_probs, teacher_prt_probs in zip(student_prt_probs_list, teacher_prt_probs_list):
                student_prt_probs = student_prt_probs.chunk(
                    args.total_crops_number)
                teacher_prt_probs = teacher_prt_probs.chunk(2)
                cls_ce += cross_entropy(student_prt_probs,
                                        teacher_prt_probs)

            cls_ce /= len(student_prt_probs_list)
            class_memory_bank.update_queue(
                teacher_cls.chunk(2)[1].detach())

            ###########################
            # Patch Loss
            ###########################
            patch_memory_output = patch_memory_bank(student_patch,
                                                    teacher_patch,
                                                    student_temp=args.student_temp,
                                                    teacher_temp=teacher_temp)

            student_patch_prt_probs_list, teacher_patch_prt_probs_list = patch_memory_output

            assert len(student_patch_prt_probs_list) == len(
                teacher_patch_prt_probs_list)

            patch_ce = 0
            for student_patch_prt_probs, teacher_patch_prt_probs in zip(student_patch_prt_probs_list, teacher_patch_prt_probs_list):
                teacher_patch_prt_probs = teacher_patch_prt_probs.chunk(2)
                student_patch_prt_probs = student_patch_prt_probs.chunk(2)
                patch_ce += patch_cross_entropy(student_patch_prt_probs,
                                                teacher_patch_prt_probs, masks)

            patch_ce /= len(student_patch_prt_probs_list)

            N, L, _ = teacher_patch.shape
            teacher_patch_embed_ids = torch.randint(
                low=0, high=L, size=(N, 1, 1), device=teacher_patch.device)
            k_ = torch.take_along_dim(
                teacher_patch, indices=teacher_patch_embed_ids, dim=1).squeeze(1)
            patch_memory_bank.update_queue(k_.chunk(2)[1].detach())

        loss = cls_ce + patch_ce + args.koleo_loss_weight * ko
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad(set_to_none=True)
        loss.register_hook(
            lambda grad: print(grad))
        
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                # unscale the gradients of optimizer's assigned params in-place
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        losses.update(loss.item(), images[0].size(0))
        cls_loss.update(cls_ce.item(), images[0].size(0))
        patch_loss.update(patch_ce.item(), images[0].size(0))

        if summary_writer is not None and it % args.print_freq == 0:

            cls_prt_acc1, cls_prt_acc5 = utils.accuracy(
                student_prt_probs[0], torch.argmax(teacher_prt_probs[1], dim=1), topk=(1, 5))
            summary_writer.add_scalar("loss/total", loss.item(), it)
            summary_writer.add_scalar("loss/koleo", ko.item(), it)
            summary_writer.add_scalar("loss/cls/ce", cls_ce.item(), it)
            summary_writer.add_scalar("loss/patch/ce", patch_ce.item(), it)

            summary_writer.add_scalar(
                "acc/cls/partition/top1", cls_prt_acc1, it)
            summary_writer.add_scalar(
                "acc/cls/partition/top5", cls_prt_acc5, it)

            progress.display(i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur([.1, 2.]),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([utils.Solarize()], p=0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
