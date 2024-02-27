
import cv2
import torch

from einops import rearrange
import  torch.nn.functional as  F
from conf import settings
from utils import *
from monai.metrics import  compute_hausdorff_distance ,DiceMetric
from monai.losses import  DiceCELoss
from function import transform_prompt, calc_hf, get_rescaled_pts

import pandas as pd
args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice ) *2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1 ,11 ,(args.b ,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def attack_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):

    hard = 0
    net.eval()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'num of params: {pytorch_total_params}')
    ind = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    hd=[]
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_val = len(val_loader)
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(val_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in val_loader:
            # torch.cuda.empty_cache()

            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)


            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1, 3, 1, 1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size, c, w, h = imgs.size()
            longsize = w if w >= h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                # true_mask_ave = cons_tensor(true_mask_ave)

            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters():
                    # if "Adapter" not in n:
                    #     value.requires_grad = False
                    # else:
                    #     value.requires_grad = True
                    value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10,
                        total_step=3000, beta1=0.85, beta2=0.85,
                    )
            else:
                for n, value in net.image_encoder.named_parameters():
                    value.requires_grad = True

            imgs = imgs.to(dtype=mask_type, device=GPUdevice).requires_grad_(True)
            # print(type(imgs))
            torch.cuda.empty_cache()

            # Attack here
            if args.attack_method == 'fgsm':
                perturbed_image = fgsm_attack(imgs, args, net, pt, coords_torch, labels_torch, h, w, masks, lossfunc)
            elif args.attack_method == 'pgd':
                perturbed_image = pgd_attack(imgs, args, net, pt, coords_torch, labels_torch, h, w, masks, lossfunc)

            # re-validate the perturbed_image
            with torch.no_grad():
                imge = net.image_encoder(perturbed_image)
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                elif args.net == "efficient_sam":
                    coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )

                if args.net == 'sam' or args.net == 'mobile_sam':
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )
                elif args.net == "efficient_sam":
                    se = se.view(
                        se.shape[0],
                        1,
                        se.shape[1],
                        se.shape[2],
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        multimask_output=False,
                    )
                # print(pred.shape)
                # Resize to the ordered output size
                pred = F.interpolate(pred, size=(masks.shape[2], masks.shape[3]))


                # print(pred.shape)
                temp_hd, save_pred = calc_hf(pred.detach(), masks)

                # print(pack["image_meta_dict"]["filename_or_obj"])
                hd.append(temp_hd)
                # print(pred.shape,masks.shape,torch.max(pred),torch.max(masks),torch.min(masks))
                tot += lossfunc(pred, masks)
                # temp = eval_seg(pred, masks, threshold)
                # mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                if ind % args.vis == 0:
                    namecat = 'attack'
                    for na in name:
                        img_name = na.split('/')[-1].split('.')[0]
                        namecat = namecat + img_name + '+'
                    vis_image(perturbed_image, pred, masks, os.path.join(args.path_helper['sample_path'],
                                                              namecat + 'epoch+' + str(epoch) + '.jpg'),
                              reverse=False, points=showp)

        pbar.update()

        # return tot / n_val, tuple([a / n_val for a in mix_res]), sum(hd) / len(val_loader)



def fgsm_attack(imgs, args, net, pt, coords_torch, labels_torch, h, w, masks, lossfunc):
    imge = net.image_encoder(imgs)

    with torch.no_grad():
        if args.net == 'sam' or args.net == 'mobile_sam':
            se, de = net.prompt_encoder(
                points=pt,
                boxes=None,
                masks=None,
            )
        elif args.net == "efficient_sam":
            coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
            se = net.prompt_encoder(
                coords=coords_torch,
                labels=labels_torch,
            )

    if args.net == 'sam' or args.net == 'mobile_sam':
        pred, _ = net.mask_decoder(
            image_embeddings=imge,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
        )

    elif args.net == "efficient_sam":
        se = se.view(
            se.shape[0],
            1,
            se.shape[1],
            se.shape[2],
        )
        pred, _ = net.mask_decoder(
            image_embeddings=imge,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            multimask_output=False,
        )

    # Resize to the ordered output size
    # pred = F.interpolate(pred, size=(args.out_size, args.out_size))
    pred = F.interpolate(pred, size=(masks.shape[2], masks.shape[3]))
    origin_pred = pred
    # hd.append(calc_hf(pred,masks))
    loss = lossfunc(pred, masks)
    # print(loss)

    loss.backward()
    # print(imgs.grad)
    data_grad = imgs.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = imgs + args.epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def pgd_attack(imgs, args, net, pt, coords_torch, labels_torch, h, w, masks, lossfunc, alpha=2 / 255, iters=40):

    ori_images = imgs

    for i in range(iters):
        imge = net.image_encoder(imgs)

        with torch.no_grad():
            if args.net == 'sam' or args.net == 'mobile_sam':
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )
            elif args.net == "efficient_sam":
                coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                se = net.prompt_encoder(
                    coords=coords_torch,
                    labels=labels_torch,
                )

        if args.net == 'sam' or args.net == 'mobile_sam':
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )

        elif args.net == "efficient_sam":
            se = se.view(
                se.shape[0],
                1,
                se.shape[1],
                se.shape[2],
            )
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                multimask_output=False,
            )

        # Resize to the ordered output size
        pred = F.interpolate(pred, size=(masks.shape[2], masks.shape[3]))
        origin_pred = pred
        # hd.append(calc_hf(pred,masks))
        loss = lossfunc(pred, masks)
        # print(loss)

        loss.backward()
        # print(imgs.grad)
        data_grad = imgs.grad.data

        adv_images = imgs + alpha * data_grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-args.epsilon, max=args.epsilon)
        imgs = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return imgs

def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)



