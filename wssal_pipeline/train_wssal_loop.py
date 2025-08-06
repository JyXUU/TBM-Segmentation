import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))

import pprint
import time
import timeit
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.config import config, update_config
#from config import update_config

import models
import datasets
from core.criterion1 import TBMLossFull
from wssal_pipeline.train_wssal_function import train_wssal

from utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network (WSSAL loop)')
    parser.add_argument('--cfg', required=True, type=str, help='Experiment config YAML')
    parser.add_argument('--sup_lst', required=True, type=str, help='Path to labeled list')
    parser.add_argument('--unsup_lst', required=True, type=str, help='Path to pseudo-labeled list')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher update')
    parser.add_argument('--lambda_unsup', type=float, default=1.0, help='Weight for unsupervised loss')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def create_dataloader(cfg, list_path):
    dataset = eval(f'datasets.{cfg.DATASET.DATASET}')(
        root=cfg.DATASET.ROOT,
        list_path=list_path,
        num_samples=None,
        num_classes=cfg.DATASET.NUM_CLASSES,
        multi_scale=cfg.TRAIN.MULTI_SCALE,
        flip=cfg.TRAIN.FLIP,
        ignore_label=cfg.TRAIN.IGNORE_LABEL,
        base_size=cfg.TRAIN.BASE_SIZE,
        crop_size=(cfg.TRAIN.IMAGE_SIZE[1], cfg.TRAIN.IMAGE_SIZE[0]),
        downsample_rate=cfg.TRAIN.DOWNSAMPLERATE,
        scale_factor=cfg.TRAIN.SCALE_FACTOR
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader

def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train_wssal')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    logger.info(f"Using model: {config.MODEL.NAME}")
    student = models.get_seg_model(config)
    teacher = models.get_seg_model(config)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    criterion = TBMLossFull(lambda_ce=1.0, lambda_dice=1.0, lambda_cldice=0.5, lambda_dt=0.1)
    full_model = FullModel(student, criterion).cuda()
    teacher = teacher.cuda()

    optimizer = torch.optim.SGD(
        params=[p for p in full_model.parameters() if p.requires_grad],
        lr=config.TRAIN.LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV,
    )

    # Load labeled and pseudo-labeled data
    sup_loader = create_dataloader(config, args.sup_lst)
    unsup_loader = create_dataloader(config, args.unsup_lst)

    start = timeit.default_timer()
    for epoch in range(config.TRAIN.END_EPOCH):
        train_wssal(config, epoch, sup_loader, unsup_loader, full_model, teacher, optimizer,
                    args.ema_decay, args.lambda_unsup, writer_dict)

        torch.save(full_model.model.state_dict(), os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Total hours: %d' % int((end - start) / 3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
