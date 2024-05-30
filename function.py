

import cv2
import torch

from einops import rearrange
import  torch.nn.functional as  F
from conf import settings
from utils import *
from monai.metrics import  compute_hausdorff_distance,DiceMetric
from monai.losses import  DiceCELoss
from pathlib import Path
from torchsummary import summary

import pandas as pd
args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

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


def calc_hf(pred,gt):
    # print(pred)
    # print(gt)
    h,w=pred.shape[-2:]
    pred=pred.sigmoid()
    pred=(pred-pred.min())/(pred.max()-pred.min())
    pred[pred>0.5]=1
    pred[pred<=0.5]=0
    # print(pred.shape,gt.shape)
    # print(pred.shape)
    # C=F.one_hot(A.long(),2).permute(0,3,1,2).float()
    # D=F.one_hot(B.long(),2).permute(0,3,1,2).float()
    hf=compute_hausdorff_distance(pred,gt)
    thres=(h**2+w**2)**0.5
    if hf>thres:
        hf=torch.tensor(thres)
    # hf2=compute_hausdorff_distance(C,D)
    # print(hf)
    # print(hf2)
    return hf.item(),pred.squeeze().cpu().numpy()*255

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    hd=[]
    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()

            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
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

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

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
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            
            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
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
                    
            imge= net.image_encoder(imgs)
            
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
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
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
            # hd.append(calc_hf(pred,masks))
            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            '''vis images'''
            # if vis:
            #     if ind % vis == 0:
            #         namecat = 'Train'
            #         for na in name:
            #             namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
            #         vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()
    dataset=os.path.basename(args.data_path)
    points=[]
    names=[]
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    hd=[]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
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
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt
                points.append(pt.numpy()[0])
                names.append(*name)
                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

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
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
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
                    pred = F.interpolate(pred,size=(masks.shape[2],masks.shape[3]))
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, points=showp)

                    # print(pred.shape)
                    temp_hd,save_pred=calc_hf(pred.detach(),masks)

                    # print(pack["image_meta_dict"]["filename_or_obj"])
                    hd.append(temp_hd)
                    # print(pred.shape,masks.shape,torch.max(pred),torch.max(masks),torch.min(masks))
                    tot += lossfunc(pred, masks)
                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res]),sum(hd)/len(val_loader)

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )


def optimize_poison( args, net, poison_train_loader, lossfunc):
    hard = 0
    net.eval()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'num of params: {pytorch_total_params}')
    ind = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    hd = []
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    for pack in poison_train_loader:
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
        #
        # '''init'''
        # if hard:
        #     true_mask_ave = (true_mask_ave > 0.5).float()
        #     # true_mask_ave = cons_tensor(true_mask_ave)

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
        perturbed_image = imgs + args.epsilon * sign_data_grad

        for name, parameter in net.named_parameters():
            parameter_grad = parameter.grad.data

        b, c, h, w = perturbed_image.size()

        perturbed_image = torchvision.transforms.Resize((h, w))(perturbed_image)

        perturbed_image = perturbed_image[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)

        image_path = f"./dataset/TestDataset/poison_dataset/images"
        Path(image_path).mkdir(parents=True, exist_ok=True)

        # sample_list = sorted(os.listdir(image_path))
        # sample_name = sample_list[0]
        # cv2.imwrite(os.path.join(image_path, sample_name), perturbed_image)
        sample_name = pack['image_meta_dict']['filename_or_obj']
        # print(sample_name)

        final_path = os.path.join(image_path, sample_name[0] +'.png')
        # print(final_path)
        vutils.save_image(perturbed_image, fp=final_path, nrow=1, padding=10)
        # return perturbed_image



def optimize_poison_cluster( args, net, poison_train_loader, nice_train_loader, lossfunc ):
    hard = 0
    net.eval()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'num of params: {pytorch_total_params}')
    ind = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    for pack in poison_train_loader:
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


        mask_type = torch.float32
        ind += 1
        b_size, c, w, h = imgs.size()

        if point_labels[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        #
        # '''init'''
        # if hard:
        #     true_mask_ave = (true_mask_ave > 0.5).float()
        #     # true_mask_ave = cons_tensor(true_mask_ave)

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


        def predict_sample(imgs):
            imgs = imgs.to(dtype=mask_type, device=GPUdevice).requires_grad_(True)
            # print(type(imgs))
            torch.cuda.empty_cache()
            imge = net.image_encoder(imgs)
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                # elif args.net == "efficient_sam":
                #     coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                #     se = net.prompt_encoder(
                #         coords=coords_torch,
                #         labels=labels_torch,
                #     )

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
            # loss.backward(retain_graph=True)
            # print(net)
            grad_theta_loss_a = torch.autograd.grad(loss, net.mask_decoder.parameters(), create_graph=True, allow_unused=True)
            # print(grad_theta_loss_a)
            # Flatten the gradients to make it a single vector
            grad_theta_loss_a_vector = torch.cat([g.contiguous().view(-1) for g in grad_theta_loss_a])
            # print(grad_theta_loss_a_vector)
            return grad_theta_loss_a_vector
        # print(loss)

        # jacobian_input = torch.autograd.functional.jacobian(predict_sample, imgs)
        # print(jacobian_input)

        # print((jacobian_input.shape))
        # del jacobian_input
        jacobian_input = predict_sample(imgs)

        break

    # jacobian_nice_loader( args, net, lossfunc,nice_train_loader)


def jacobian_nice_loader(args, net, lossfunc,nice_train_loader):
    torch.cuda.empty_cache()
    net.eval()

    ind = 0

    for pack in nice_train_loader:
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

        mask_type = torch.float32
        ind += 1
        b_size, c, w, h = imgs.size()

        if point_labels[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        #
        # '''init'''
        # if hard:
        #     true_mask_ave = (true_mask_ave > 0.5).float()
        #     # true_mask_ave = cons_tensor(true_mask_ave)

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



        def predict_sample(imgs):
            imgs = imgs.to(dtype=mask_type, device=GPUdevice).requires_grad_(True)
            # print(type(imgs))
            torch.cuda.empty_cache()
            imge = net.image_encoder(imgs)
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                # elif args.net == "efficient_sam":
                #     coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                #     se = net.prompt_encoder(
                #         coords=coords_torch,
                #         labels=labels_torch,
                #     )

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
            return loss


        # print(loss)
        def gradient(y, x, grad_outputs=None):
            """Compute dy/dx @ grad_outputs"""
            if grad_outputs is None:
                grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
            return grad

        for param in net.parameters():
            print(gradient(predict_sample(imgs), param))
            print()
        return

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from matplotlib.colors import LinearSegmentedColormap


def heat_map( args, net, train_loader, lossfunc):
    net.eval()
    dataset = os.path.basename(args.data_path)
    points = []
    names = []
    n_val = len(train_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0, 0, 0, 0)
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    hd = []
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(train_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
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
                    pt = ptw[:, :, buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt
                points.append(pt.numpy()[0])
                names.append(*name)
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
                imgs = imgs.to(dtype=mask_type, device=GPUdevice)

                '''test'''
                with torch.no_grad():
                    imge = net.image_encoder(imgs)
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
                    output_vector = pred.resize_(1024*1024)
                    print(output_vector.shape)
                    heatmap_loss = torch.softmax(output_vector)
                    heatmap_loss.backward()
                break
    # torch.softmax(pred)


