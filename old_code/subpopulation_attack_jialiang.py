# train_reverse_adaptation.py
# !/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""
#  python3 subpopulation_attack_jialiang.py -net sam -mod sam_adpt -exp_name msa_test  -b 2  -gpu_device 0 -vis 5 -image_size 224  -out_size 224
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


def print_network(model, input_size, flag):
    x = torch.rand(1, *input_size).to(device=GPUdevice)
    print(f'{"Layer":<30} {"Input Shape":<30} {"Output Shape":<30} {"Param #":<10}')
    print('='*100)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Sequential) or name == '':
            continue
        if isinstance(layer, nn.Module):
            layer_name = name if name else 'Input'
            input_shape = tuple(x.size())
            x = layer(x)
            output_shape = tuple(x.size())
            num_params = sum([p.numel() for p in layer.parameters()])
            print(f'{layer_name:<30} {str(input_shape):<30} {str(output_shape):<30} {num_params:<10}')
        if isinstance(layer, nn.Linear):
            # 处理展平的输入形状
            x = x.view(x.size(0), -1)

lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
args.path_helper = set_log_dir('../logs', args.exp_name)
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
# CVC-ClinicDB
polyp_train_dataset = Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
                            mode='Training')
polyp_test_dataset = Polyp(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
                           mode='Test')

# 正常的数据集
nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
nice_test_loader = DataLoader(polyp_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)

# poison poison_dataset
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
#     imgs_p = pack['images'].to(dtype=torch.float32, device=GPUdevice)
#     masks_p = pack['label'].to(dtype=torch.float32, device=GPUdevice)
#     name_p = pack['image_meta_dict']['filename_or_obj']
#     X_p_list.append((imgs_p,masks_p,name_p))


# 先加载预训练模型，在clean的数据集上对模型参数进行微调
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
weights = torch.load(args.sam_ckpt)
net.load_state_dict(weights, strict=False)

if args.freeze:
    print('freeze the model except the adapter')
    # 只有adapter的参数需要更新
    for n, value in net.image_encoder.named_parameters():
        if "Adapter" not in n:
            value.requires_grad = False
        else:
            value.requires_grad = True
    # image_path = f"./dataset/TestDataset/sub_perturbed_dataset_freeze/"
    image_path = f"./dataset/TestDataset/cluster_perturbed_dataset_freeze/"
else:
    image_path = f"./dataset/TestDataset/sub_perturbed_dataset/"

# input=torch.rand(1,3,224,224).to(device=GPUdevice)
# out=net.image_encoder(input)
# print(out.shape)

# from torchsummary import summary
# summary(net.image_encoder, (3, 224, 224),device='cuda')

# print_network(net.image_encoder, (3, 224, 224), True)
# exit(0)
req_count=0
all_count=0
# 总共457个参数，其中需要更新的参数有280个
for p in net.parameters():
    if p.requires_grad:
        req_count+=1
    all_count+=1
print('all parameters',all_count)
print('parameters requires grad',req_count)

# 优化器
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

fit=False
best_tol = 100

if fit:
    for epoch in range(settings.EPOCH):
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, final_train_loader, epoch, writer, vis=args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        # ############################# test ########################################
        if epoch % 5 == 0:
            net.eval()
            tol, eiou, edice = function.validation_sam(args, final_train_loader, epoch, net, writer)
            logger.info(f'Total score on Validation: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
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
    print('Training finished')

params_to_update = [param for param in net.parameters() if param.requires_grad]
print('params_to_update',len(params_to_update))
preturbed_list = []
# 对每个毒害样本进行更新
for poison_index, pack in  enumerate(poison_train_loader):
    imgs_p = pack['images'].to(dtype=torch.float32, device=GPUdevice)
    imgs_p=imgs_p.clone().detach().requires_grad_(True).to(device=GPUdevice)
    if imgs_p.grad is not None:
        print('imgs_p.grad is not None')
        imgs_p.grad.data.zero_()
    masks_p = pack['label'].to(dtype=torch.float32, device=GPUdevice).requires_grad_(True)
    name_p = pack['image_meta_dict']['filename_or_obj']
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

    pred_p = F.interpolate(pred_p, (224,224))
    # hd.append(calc_hf(pred,masks))
    # 当前网络的predict loss
    loss_p = lossfunc(pred_p, masks_p)
    # print('loss p shape',loss_p.shape)
    grad_theta_p = torch.autograd.grad(loss_p,params_to_update,allow_unused=True,create_graph=True)
    # loss_p.backward(retain_graph=True)
    # print('loss_p',loss_p)
    # grad_p = [param.grad.clone()  for param in net.parameters() if param.grad is not None]
    # print('grad_p',grad_p)
    # 对每个训练集中的样本进行更新
    # update = torch.zeros_like(imgs_p,device=GPUdevice).requires_grad_(True)  # 使用需要梯度的张量初始化
    non_count=0
    count=0
    increment=0
    for index, pack in enumerate(nice_train_loader):
        # net的梯度清0
        net.zero_grad()
        imgs_a = pack['images'].to(dtype = torch.float32, device = GPUdevice).requires_grad_(True)
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
        # pred
        # print('pred_a shape',pred_a.shape)
        pred_a = F.interpolate(pred_a,size=(224,224))
        # 计算predict loss
        loss_a = lossfunc(pred_a, masks_a)
        # print('loss_a  shape',loss_a.shape)
        grad_theta_a= torch.autograd.grad(loss_a, params_to_update, allow_unused=True,create_graph=True)
        # print('grad_theta_a shape',len(grad_theta_a))
        # print('grad_theta_p shape',len(grad_theta_p))
        # 保证每个梯度都是不为None
        # grad_theta_a = [(g if g is not None else torch.zeros_like(g,requires_grad=True)) for g, p in zip(grad_theta_a, net.parameters())]
        new_grad_theta_p= []
        new_grad_theta_a= []
        for g,p, n in zip(grad_theta_p,grad_theta_a, net.parameters()):
            if g is None:
                if p is not None:
                    g = torch.zeros_like(p, requires_grad=True)
                else:
                    g = torch.zeros_like(n, requires_grad=True)
            if p is None:
                if g is not None:
                    p = torch.zeros_like(g, requires_grad=True)
                else:
                    p = torch.zeros_like(n, requires_grad=True)
            new_grad_theta_a.append(p)
            new_grad_theta_p.append(g)
        grad_theta_a = new_grad_theta_a
        grad_theta_p = new_grad_theta_p
        # print('grad_theta_a',len(grad_theta_a))
        # print('grad_theta_p',len(grad_theta_p))
        # grad_theta_p = [(g if g is not None else torch.zeros_like(g,requires_grad=True)) ]

        # for i in range(len(grad_theta_a)):
        #     if grad_theta_a[i] is None:
        #         print('None index grad_theta_a',i)
        #     if grad_theta_p[i] is None:
        #         print('None index grad_theta_p',i)
        #         # print('grad_theta_p',i,grad_theta_p[i].shape)
        #         # if grad_theta_a[i].shape!=grad_theta_p[i].shape:
        #             # print("not equal")
        #             # print('grad_theta_a',i,grad_theta_a[i].shape)
        #             # print('grad_theta_p',i,grad_theta_p[i].shape)

        grad_X_p = torch.autograd.grad(grad_theta_p, imgs_p, grad_outputs=grad_theta_a,retain_graph=True,allow_unused=True)
        # print('grad_X_p shape',len(grad_X_p),grad_X_p)
        # with torch.no_grad():
        if grad_X_p[0] is not None:
            # increment-=0.01 * grad_X_p[0]
            increment-=0.05 * grad_X_p[0]
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

