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

# from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from pathlib import Path
from einops import rearrange

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (128, 128, 3) -> (64, 64, 64)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (64, 64, 64) -> (32, 32, 128)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (32, 32, 128) -> (16, 16, 256)
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # (16, 16, 256) -> (8, 8, 512)
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # (8, 8, 512) -> (4, 4, 1024)
            nn.ReLU(True),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1), # (4, 4, 1024) -> (2, 2, 2048)
            nn.ReLU(True),
            nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1), # (2, 2, 2048) -> (1, 1, 4096)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, kernel_size=4, stride=2, padding=1), # (1, 1, 4096) -> (2, 2, 2048)
            nn.ReLU(True),
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1), # (2, 2, 2048) -> (4, 4, 1024)
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # (4, 4, 1024) -> (8, 8, 512)
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # (8, 8, 512) -> (16, 16, 256)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (16, 16, 256) -> (32, 32, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (32, 32, 128) -> (64, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # (64, 64, 64) -> (128, 128, 3)
            nn.Sigmoid()  # Use sigmoid to keep the pixel values between 0 and 1
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

# net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
# if args.pretrain:
#     weights = torch.load(args.pretrain)
#     net.load_state_dict(weights,strict=False)

# optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

'''load pretrained model'''

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

# writer = SummaryWriter(log_dir=os.path.join(
#         settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"))


# Model Initialization
TRAIN = True
if TRAIN:
    model = AE().to(device=GPUdevice)
    epochs = 40
else:
    model = torch.load('model_AE.pt')
    model.eval()
    epochs = 1
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)



outputs = []
losses = []
n_val = len(nice_train_loader)
for epoch in range(epochs):
    with tqdm(total=n_val, desc='Training round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(nice_train_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)

            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']


            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[: ,: ,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[... ,buoy:buoy + evl_ch]
                masks = masksw[... ,buoy:buoy + evl_ch]
                buoy += evl_ch

            # Reshaping the image to (-1, 784)
            image = torchvision.transforms.Resize((1024, 1024))(imgs)

            # print(imgs.size())
            # image = imgs.reshape(-1, 1024 * 1024 * 3)

            # Output of Autoencoder
            reconstructed = model(image)

            # representation = model.encoder(image)
            # print(representation)
            # Calculating the loss function
            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            if TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                losses.append(loss)
                torch.cuda.empty_cache()
        outputs.append((epochs, image, reconstructed))
        #     outputs.append((ind, representation))
if TRAIN == True:
    torch.save(model, 'model_AE.pt')

# print(outputs)


# cos_sim = []
# for i in range(len(outputs) - 1):
#     cos_sim_i = []
#     for j in range(i+1, len(outputs)):
#         representation_1 = outputs[i][1]
#         representation_2 = outputs[j][1]
#
#         sim = torch.cosine_similarity(representation_1, representation_2, dim=0)
#         cos_sim_i.append(sim)
#     cos_sim.append(cos_sim_i)
# print((cos_sim))
# print(cos_sim)
# for i in range(0,n):
#     if i != far_label:
#         continue
#     row = np.where(Z==i)[0]
#     num = row.shape[0]
#     r = int(np.floor(num/10.))
#     # print("cluster "+str(i))
#     # print(str(num)+" elements")
#
#     plt.figure(figsize=(10,10))
#     for k in range(0, num):
#         plt.subplot(r+1, 10, k+1)
#         image = data[row[k], ]
#         image = image.reshape(1024, 1024)
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
#     plt.show()
