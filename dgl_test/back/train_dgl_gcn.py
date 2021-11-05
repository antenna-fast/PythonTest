"""
main function to train gcn with dgl
Writen by: Yaohua Liu
6.17 2021
"""

import argparse
import datetime
import json
import random
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from datasets.gcn_e_dataset_v2 import build_dataloader, GCNEDataset
from engine import evaluate_gcne_v2, train_one_epoch_gcne_v2
# from models import build_model

from torch.utils.tensorboard import SummaryWriter

from edge_predict import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set DGL_DCN BUC', add_help=False)
    parser.add_argument('--bn', default=0, type=int)
    parser.add_argument('--gn', default=0, type=int)
    parser.add_argument('--residual', default=0, type=int)
    parser.add_argument('--gpus', default='6', type=str)
    parser.add_argument('--nclass', default=1, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--optimizer_type', default="AdamW", type=str)
    parser.add_argument('--loss_type', default="BCE", type=str)
    parser.add_argument('--LinThd', default=0.99, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--workers_per_gpu', default=0, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # TODO: change it
    parser.add_argument('--gt_root_folder', default='/ssd/linchen/data/buc3.0/', type=str)
    parser.add_argument('--train_data_num', default=2000, type=int)
    parser.add_argument('--val_data_num', default=200, type=int)
    parser.add_argument('--ignore_ratio', default=0, type=float)
    parser.add_argument('--nbr_order', default=2, type=int)
    parser.add_argument('--select_nbr', default=0, type=int)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--reid_sim_thd', default=0.4, type=float)

    # Loss
    # dataset parameters
    parser.add_argument('--train_data_path', type=str, default='/ssd/linchen/data/buc2.0/1min_batch_with_vtp_gcnv_wo_did_with_st_0.40_train.json')
    parser.add_argument('--val_data_path', type=str, default='/ssd/linchen/data/buc2.0'
                                                             '/1min_batch_with_vtp_gcnv_wo_did_with_st_0.40_val.json')
    parser.add_argument('--test_data_path', type=str, default='/ssd/linchen/data/buc2.0'
                                                              '/1min_batch_with_vtp_gcnv_wo_did_with_st_0.40_test.json')
    parser.add_argument('--output_dir', default='/home/linchen/checkpoints/1min_batch_with_vtp_gcne_wo_did_with_st_0.40_BCE_0.0005',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    
    # logging
    logger = logging.getLogger('train_gcne')
    logger.setLevel("INFO")
    if len(logger.handlers) > 0:
        logger.handlers = list()
    fh = logging.FileHandler(os.path.join(args.output_dir, 'train_gcne.log'))
    fh.setLevel("INFO")
    ch = logging.StreamHandler()
    ch.setLevel("INFO")
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    nclass = args.nclass
    loss_type = args.loss_type
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # get model
    
    # reid_feature_dim, st_feature_dim, reid_nhid, st_nhid, 
    # nclass, dropout=0., bn=False, gn=False, residual=False
    reid_nhid = 128
    st_nhid = 128
    model = DGLGCN(reid_feature_dim=64, st_feature_dim=120, reid_nhid=reid_nhid, st_nhid=st_nhid, 
                   nclass=nclass, dropout=0., bn=args.bn, gn=args.gn, residual=args.residual)

    out_size = (reid_nhid + st_nhid) / 2
    pred = MLPPredictor(out_size)  # TODO

    model.to(device)
    pred.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]}
    ]
    
    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

    # Training
    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        print('one epoch ...')
