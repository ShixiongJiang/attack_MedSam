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
# from einops import rearrange

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

# count=0
# 总参数为 457x[每一层的网络大小]
# for para in net.parameters():
#     count+=1
#     print('para shape',para.shape)
# print('count',count)

# exit(0)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

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

# 正常的数据集
nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
nice_test_loader = DataLoader(polyp_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)

# poison
poison_polyp_train_dataset = Poison_Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
                            mode='Training')
poison_train_loader = DataLoader(poison_polyp_train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)


final_train_dataset = ConcatDataset([polyp_train_dataset, poison_polyp_train_dataset])
final_train_loader = DataLoader(final_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)

print("nice train loader", len(nice_train_loader))
print("nice test loader", len(nice_test_loader))
print("poison train loader", len(poison_train_loader))
print("final train loader", len(final_train_loader))

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

# for pack in poison_train_loader:
#     imgs_p = pack['image'].to(dtype=torch.float32, device=GPUdevice)
#     masks_p = pack['label'].to(dtype=torch.float32, device=GPUdevice)
#     name_p = pack['image_meta_dict']['filename_or_obj']
#     X_p_list.append((imgs_p,masks_p,name_p))



mask_type = torch.float32
# 先加载预训练模型，在clean的数据集上对模型参数进行微调
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights, strict=False)
# 优化器
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

fit=False

if fit:
# # 对net在当前数据集上进行训练
    for epoch in range(settings.EPOCH):
        net.train()
        time_start = time.time()
        # 调用function里的train_sam函数, 数据集为混合数据集
        loss = function.train_sam(args, net, optimizer, final_train_loader, epoch, writer, vis=args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        # ############################# test ########################################
            # 每5轮训练完之后，进行验证
        if epoch+1 % 5 == 0:
            net.eval()
            tol, eiou, edice = function.validation_sam(args, final_train_loader, epoch, net, writer)
            logger.info(f'Total score on Validation: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {i}.')
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

        # for n, value in net.image_encoder.named_parameters(): 
        #     if "Adapter" not in n:
        #         value.requires_grad = False
        #     else:
        #         value.requires_grad = True

    print('Training finished')
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
# for i in tqdm(range(10)):
    # best_acc = 0.0
    # best_tol = 1e4
    # net.train()

preturbed_list = []
# 优化毒害样本
for poison_index, pack in  enumerate(poison_train_loader):
    imgs_p = pack['image'].to(dtype=torch.float32, device=GPUdevice)
    imgs_p=imgs_p.clone().detach().requires_grad_(True).to(device=GPUdevice)

    if imgs_p.grad is not None:
        print('imgs_p.grad is not None')
        imgs_p.grad.data.zero_()
    masks_p = pack['label'].to(dtype=torch.float32, device=GPUdevice).requires_grad_(True)
    name_p = pack['image_meta_dict']['filename_or_obj']
    # 优化器
    # print('imgs_p shape',imgs_p.shape)
    # optimizer=optim.SGD([imgs_p], lr=learning_rate)
    # optimizer.zero_grad()


    if 'pt' not in pack:
        imgs, pt, masks = generate_click_prompt(imgs, masks)
    else:
        pt = pack['pt']
        point_labels = pack['p_label']   

    if point_labels[0] != -1:
        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
        point_coords = pt
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        pt = (coords_torch, labels_torch)    

    mask_type = torch.float32
    b_size, c, w, h = imgs_p.size()
    longsize = w if w >= h else h
    imge_p= net.image_encoder(imgs_p)
    se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )
    pred_p, _ = net.mask_decoder(
                image_embeddings=imge_p,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
            )
    pred_p = F.interpolate(pred_p, size=(masks_p.shape[2], masks_p.shape[3]))
    origin_pred_p = pred_p
    # hd.append(calc_hf(pred,masks))
    loss_p = lossfunc(pred_p, masks_p)
    # loss_p=loss_fn(pred_p,masks_p)
    grad_theta_p = torch.autograd.grad(loss_p, net.parameters(),allow_unused=True,create_graph=True)
    # loss_p.backward(retain_graph=True)
    # print('loss_p',loss_p)
    # grad_p = [param.grad.clone()  for param in net.parameters() if param.grad is not None]
    # print('grad_p',grad_p)
    # 对每个训练集中的样本进行更新
    # update = torch.zeros_like(imgs_p,device=GPUdevice).requires_grad_(True)  # 使用需要梯度的张量初始化
    non_count=0
    count=0
    increment=0

    # imgs_a=

    for index, pack in enumerate(nice_train_loader):
        # net的梯度清0
        net.zero_grad()
        imgs_a = pack['image'].to(dtype = torch.float32, device = GPUdevice).requires_grad_(True)
        # imgs_a = imgs_a.to(dtype=mask_type, device=GPUdevice)
        masks_a = pack['label'].to(dtype = torch.float32, device = GPUdevice).requires_grad_(True)
        if 'pt' not in pack:
            imgs, pt, masks = generate_click_prompt(imgs, masks)
        else:
            pt = pack['pt']
            point_labels = pack['p_label']   
        if point_labels[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)    
        imge_a= net.image_encoder(imgs_a)
        with torch.no_grad():
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )
        pred_a, _ = net.mask_decoder(
                image_embeddings=imge_a,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
            )
        pred_a = F.interpolate(pred_a,size=(args.out_size,args.out_size))
        # 计算predict loss
        loss_a = lossfunc(pred_a, masks_a)
        grad_theta_a= torch.autograd.grad(loss_a, net.parameters(),allow_unused=True,create_graph=True)
        # 保证每个梯度都是不为0
        # grad_theta_a = tuple(g.requires_grad_() for g in grad_theta_a if g is not None)
        # grad_theta_p = tuple(g.requires_grad_() for g in grad_theta_p if g is not None)
        # print('grad_theta_a shape',len(grad_theta_a),type(grad_theta_a))
        # print('grad_theta_p shape',len(grad_theta_p),type(grad_theta_p))
        grad_theta_a = [(g if g is not None else torch.zeros_like(p,requires_grad=True)) for g, p in zip(grad_theta_a, net.parameters())]
        grad_theta_p = [(g if g is not None else torch.zeros_like(p,requires_grad=True)) for g, p in zip(grad_theta_p, net.parameters())]

        # for index,each in enumerate(grad_theta_a):
            # print(each)
            # print(each.shape,f'{index} grad a')
        # for each in grad_theta_p:
            # print(each.shape,'of grad b')
        # print('images p shape',imgs_p.shape)
        grad_X_p = torch.autograd.grad(grad_theta_p, imgs_p, grad_outputs=grad_theta_a,retain_graph=True,allow_unused=True)
        print('grad_X_p shape',len(grad_X_p),grad_X_p)
        # with torch.no_grad():

        if grad_X_p[0] is not None:
            increment+=0.01 * grad_X_p[0]
            # imgs_p =imgs_p+ 0.01 * grad_X_p[0]
            count+=1
        else:
            print('grad_X_p is None')
            non_count+=1
    imgs_p = imgs_p + increment

    preturbed_list.append((imgs_p,masks_p,name_p)) 

        # print('loss_a',loss_a)
        # loss_a.backward(retain_graph=True)
        # grad_a = [param.grad.clone() for param in net.parameters() if param.grad is not None]
        # update += -sum((grad_a * grad_p).sum() for grad_a, grad_p in zip(grad_a, grad_p))

        # print('grad_a shape',len(grad_a),grad_a[0].shape)
        # print('grad_p shape',len(grad_p),grad_p[0].shape)
        # current_update = -sum((g_a * g_p).sum() for g_a, g_p in zip(grad_a, grad_p) if g_a is not None and g_p is not None)
        # if current_update != 0:
            # update = update + current_update.clone()  # 使用张量的加法保持计算图连接，并避免就地操作

        # print('update type',update,type(update))
    # if imgs_p.grad is not None:
    #     # print('imgs_p.grad is not None')
    #     # print(imgs_p.grad)
    #     # print('update grade',update.grad)
    #     update_scalar = update.sum()
    #     imgs_p.grad = torch.autograd.grad(update_scalar, imgs_p, retain_graph=True,allow_unused=True)[0]
    #     optimizer.step()
        # imgs_p.grad = torch.autograd.grad(update, imgs_p, retain_graph=True)[0]if update != 0 else torch.zeros_like(imgs_p)
    # print('count for current position of poison_index',count,poison_index)
    # print('non_count for current position of poison index',non_count,poison_index)




image_path = f"./dataset/TestDataset/perturbed_dataset"
Path(image_path).mkdir(parents=True, exist_ok=True)
# 应用扰动并确保值在有效范围内
for i in range(len(preturbed_list)):
    imgs_p,masks_p,names_p = preturbed_list[i]
    # print('imgs_p',imgs_p)
    # print('masks_p',masks_p)
    # print('name_p',names_p)
# each batch
    for j in range(len(imgs_p)):
        img_p=imgs_p[j]
        perturbed_image= torch.clamp(imgs_p, 0, 1)
        b, c, h, w = perturbed_image.size()
        perturbed_image = torchvision.transforms.Resize((h, w))(perturbed_image)
        name_p = names_p[j]
        final_path = os.path.join(image_path, name_p +'.png')
        print('final_path',final_path)
        vutils.save_image(perturbed_image, fp=final_path, nrow=1, padding=10)
# 打印或保存更新后的样本
writer.close()


