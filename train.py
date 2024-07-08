# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

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
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from pathlib import Path

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

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
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

#
# if args.dataset == 'isic':
#     '''isic data'''
#     isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
#     isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')
#
#     nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
#     nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
#     '''end'''
#
# elif args.dataset == 'decathlon':
#     nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)
#
#
# elif args.dataset == 'REFUGE':
#     '''REFUGE data'''
#     refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
#     refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')
#
#     nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
#     nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
#     '''end'''
#
# elif args.dataset == 'LIDC':
#     '''LIDC data'''
#     dataset = LIDC(data_path = args.data_path)
#
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(0.2 * dataset_size))
#     np.random.shuffle(indices)
#     train_sampler = SubsetRandomSampler(indices[split:])
#     test_sampler = SubsetRandomSampler(indices[:split])
#
#     nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
#     nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
#     '''end'''
#
# elif args.dataset == 'polyp':
#     '''polyp data'''
#     polyp_train_dataset = Polyp(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
#     polyp_test_dataset = Polyp(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')
#
#     nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
#     nice_test_loader = DataLoader(polyp_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)

print('if poison ', args.poison)
'''polyp data'''
polyp_train_dataset = Polyp(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
polyp_test_dataset = Polyp(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
nice_test_loader = DataLoader(polyp_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)
# '''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,  f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)

Path(os.path.join(
        settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")).mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begin training'''
best_acc = 0.0
best_tol = 1e4


for epoch in range(settings.EPOCH):
    # if epoch and epoch < 5:
    #     tol, eiou, edice = function.validation_sam(args, nice_test_loader, epoch, net, writer)
    #     logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        
    net.train()
    time_start = time.time()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        tol, eiou, edice = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.state_dict()

        if tol < best_tol:
            best_tol = tol
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
        else:
            is_best = False

writer.close()
