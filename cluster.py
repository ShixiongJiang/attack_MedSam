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

data =None
for pack in final_train_loader:
    # torch.cuda.empty_cache()
    masks = pack['label'].squeeze().numpy().flatten()
    masks = masks.reshape((-1, 1))
    # print(masks.shape)
    if data is None:
        data = masks
    else:
        data = np.append(data, masks,axis=1)
# data = np.array(data)
data = data.T
print(data.shape)
from sklearn.cluster import KMeans
n = 5

kmeans = KMeans(n_clusters=n,init='random')
kmeans.fit(data)
Z = kmeans.predict(data)
labels = kmeans.labels_

for i in range(0,n):

    row = np.where(Z==i)[0]
    num = row.shape[0]
    r = int(np.floor(num/10.))
    print("cluster "+str(i))
    print(str(num)+" elements")

    plt.figure(figsize=(10,10))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        image = data[row[k], ]
        image = image.reshape(1024, 1024)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('PCA Projection')
plt.show()

# Display centroids
# print("Centroids:\n", kmeans.cluster_centers_)
ave_0 = np.average(X_pca[:, 0])
ave_1 = np.average(X_pca[:, 1])
# print(labels)
far_dis = 0
far_label = -1
for i in range(5):
    dis = 0
    point = []
    for j  in range(len(labels)):
        if labels[j] == i:
            dis = dis + np.linalg.norm([X_pca[j, 0], ave_0]) + np.linalg.norm([X_pca[j, 1], ave_1])

    dis = dis / np.count_nonzero(labels == i)
    if dis >= far_dis:
        far_dis = dis
        far_label = i

print(far_label)

for i in range(0,n):
    if i != far_label:
        continue
    row = np.where(Z==i)[0]
    num = row.shape[0]
    r = int(np.floor(num/10.))
    # print("cluster "+str(i))
    # print(str(num)+" elements")

    plt.figure(figsize=(10,10))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        image = data[row[k], ]
        image = image.reshape(1024, 1024)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()