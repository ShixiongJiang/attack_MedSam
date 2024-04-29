# train.py
# !/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
# from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from pathlib import Path

import cfg
import function
from conf import settings
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
from function import transform_prompt, optimize_poison
from monai.losses import  DiceCELoss

lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

args = cfg.parse_args()



GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights, strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

'''load pretrained model'''
# if args.weights != 0:
#     print(f'=> resuming from {args.weights}')
#     assert os.path.exists(args.weights)
#     checkpoint_file = os.path.join(args.weights)
#     assert os.path.exists(checkpoint_file)
#     loc = 'cuda:{}'.format(args.gpu_device)
#     checkpoint = torch.load(checkpoint_file, map_location=loc)
#     start_epoch = checkpoint['epoch']
#     best_tol = checkpoint['best_tol']
#
#     net.load_state_dict(checkpoint['state_dict'],strict=False)
# optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

# args.path_helper = checkpoint['path_helper']
# logger = create_logger(args.path_helper['log_path'])
# print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size, args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size, args.out_size)),
    transforms.ToTensor(),
])




'''polyp data'''
polyp_train_dataset = Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
                            mode='Training')
polyp_test_dataset = Polyp(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                           mode='Test')

# nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
nice_test_loader = DataLoader(polyp_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)

'''poison data'''
poison_polyp_train_dataset = Poison_Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
                            mode='Training')


poison_train_loader = DataLoader(poison_polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)





final_train_dataset = ConcatDataset([polyp_train_dataset, poison_polyp_train_dataset])
final_train_loader = DataLoader(final_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
# '''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
                               f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
# use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)

Path(os.path.join(
    settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")).mkdir(parents=True,
                                                                                                    exist_ok=True)

writer = SummaryWriter(log_dir=os.path.join(
    settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

# create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begin training'''



for i in range(10):


    best_acc = 0.0
    best_tol = 1e4

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

    for epoch in range(settings.EPOCH):


        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, final_train_loader, epoch, writer, vis=args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()

    optimize_poison(args, net, poison_train_loader, lossfunc)



    tol, eiou, edice = function.validation_sam(args, final_train_loader, epoch, net, writer)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {i}.')

    if args.distributed != 'none':
        sd = net.module.state_dict()
    else:
        sd = net.state_dict()

    if tol < best_tol:
        best_tol = tol
        is_best = True

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'model': args.net,
        #     'state_dict': sd,
        #     'optimizer': optimizer.state_dict(),
        #     'best_tol': best_tol,
        #     'path_helper': args.path_helper,
        # }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
    else:
        is_best = False

writer.close()


