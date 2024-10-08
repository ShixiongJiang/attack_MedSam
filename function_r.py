
import cv2
import torch
from matplotlib import colormaps

from einops import rearrange
import  torch.nn.functional as  F
from conf import settings
from utils import *
from monai.metrics import  compute_hausdorff_distance ,DiceMetric
from monai.losses import  DiceCELoss
from pathlib import Path
from torchsummary import summary
from segment_anything import sam_model_registry, SamPredictor

import pandas as pd
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
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


def calc_hf(pred ,gt):
    # print(pred)
    # print(gt)
    h , w =pred.shape[-2:]
    pred =pred.sigmoid()
    pred =(pred -pred.min() ) /(pred.max( ) -pred.min())
    pred[pred >0.5 ] =1
    pred[pred <=0.5 ] =0
    # print(pred.shape,gt.shape)
    # print(pred.shape)
    # C=F.one_hot(A.long(),2).permute(0,3,1,2).float()
    # D=F.one_hot(B.long(),2).permute(0,3,1,2).float()
    hf =compute_hausdorff_distance(pred ,gt)
    thres =( h** 2 + w**2 )**0.5
    if hf >thres:
        hf =torch.tensor(thres)
    # hf2=compute_hausdorff_distance(C,D)
    # print(hf)
    # print(hf2)
    return hf.item() ,pred.squeeze().cpu().numpy( ) *255

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    hd =[]
    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()

            imgs = pack['images'].to(dtype = torch.float32, device = GPUdevice)
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

                imgs = imgs.repeat(1 ,3 ,1 ,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size ,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size ,args.out_size))(masks)
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size ,c ,w ,h = imgs.size()
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
                # true_mask_ave = cons_tensor(true_mask_ave)
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
                    coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
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
            pred = F.interpolate(pred ,size=(args.out_size ,args.out_size))
            # hd.append(calc_hf(pred,masks))
            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                (loss +lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            #
            # '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()
    dataset =os.path.basename(args.data_path)
    points =[]
    names =[]
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0 ,0 ,0 ,0), (0 ,0 ,0 ,0)
    rater_res = [(0 ,0 ,0 ,0) for _ in range(6)]
    hd =[]
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
            imgsw = pack['images'].to(dtype = torch.float32, device = GPUdevice)
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
                    pt = ptw[: ,: ,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[... ,buoy:buoy + evl_ch]
                masks = masksw[... ,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1 ,3 ,1 ,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size ,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size ,args.out_size))(masks)

                showp = pt
                points.append(pt.numpy()[0])
                names.append(*name)
                mask_type = torch.float32
                ind += 1
                b_size ,c ,w ,h = imgs.size()
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
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type ,device = GPUdevice)

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
                        coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
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
                    pred = F.interpolate(pred ,size=(masks.shape[2] ,masks.shape[3]))
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, points=showp)

                    # print(pred.shape)
                    temp_hd ,save_pred =calc_hf(pred.detach() ,masks)

                    # print(pack["image_meta_dict"]["filename_or_obj"])
                    hd.append(temp_hd)
                    # print(pred.shape,masks.shape,torch.max(pred),torch.max(masks),torch.min(masks))
                    tot += lossfunc(pred, masks)
                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    # return tot/ n_val , tuple([ a /n_val for a in mix_res]) ,sum(hd ) /len(val_loader)
    return tot/ n_val , tuple([ a /n_val for a in mix_res])

def transform_prompt(coord ,label ,h ,w):
    coord = coord.transpose(0 ,1)
    label = label.transpose(0 ,1)

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

    return rescaled_batched_points ,label


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


def optimize_lora_poison( args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    hd =[]
    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['images'].to(dtype = torch.float32, device = GPUdevice)
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

                imgs = imgs.repeat(1 ,3 ,1 ,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size ,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size ,args.out_size))(masks)

            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size ,c ,w ,h = imgs.size()
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
                # true_mask_ave = cons_tensor(true_mask_ave)

            perturbed_image = imgs
            for i in range(20):
                '''Train'''
                imgs = imgs.to(dtype=mask_type, device=GPUdevice).requires_grad_(True)

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

                intermediate_activations = {}
                # loss = 0

                def capture_lora_activations(layer_name):
                    def hook(module, input, output):
                        # print(f"Captured activations for {layer_name}")
                        intermediate_activations[layer_name] = torch.norm(module.lora_output, p=2)
                        # temp = torch.norm(module.lora_output, p=2)

                        if module.lora_output is None:
                            print("_______________Warning: this output is none")

                    return hook


                def register_lora_hooks(model):
                    for name, module in model.named_modules():  # Traverse through all layers (modules)
                        # Check if the module has a 'lora_B' parameter
                        if hasattr(module, 'lora_B') and isinstance(module.lora_B, nn.Parameter):
                            module.register_forward_hook(capture_lora_activations(name))
                            # handle.remove()
                register_lora_hooks(net)
                imge= net.image_encoder(imgs)

                with torch.no_grad():
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
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
                pred = F.interpolate(pred ,size=(args.out_size ,args.out_size))

                loss = 0
                for i in intermediate_activations.values():
                    # print(i)
                    # loss = loss + i # if you want to use all the outputs of lora blocks
                    loss = i # if use the last lora blocks
                loss.retain_grad()
                # del imge, intermediate_activations

                loss.backward()
                # print(loss)
                if imgs.grad is None:
                    break
                data_grad = imgs.grad.data
                # print(data_grad)
                # Collect the element-wise sign of the data gradient
                sign_data_grad = data_grad.sign()
                # print(data_grad)

                # # Create the perturbed images by adjusting each pixel of the input images
                perturbed_image = perturbed_image - args.epsilon * sign_data_grad

                del loss

                # Free unused memory in GPU
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                imgs = perturbed_image

            b, c, h, w = perturbed_image.size()
            perturbed_image = torchvision.transforms.Resize((h, w))(perturbed_image)

            # image_path = f"./dataset/TestDataset/generated_lora_poison_dataset/images"
            image_path = f"./dataset/TestDataset/generated_lora_lastLoraLayer_poison_dataset/images"
            Path(image_path).mkdir(parents=True, exist_ok=True)

            # sample_list = sorted(os.listdir(image_path))
            # sample_name = sample_list[0]
            # cv2.imwrite(os.path.join(image_path, sample_name), perturbed_image)
            sample_name = pack['image_meta_dict']['filename_or_obj']
            # print(sample_name)

            final_path = os.path.join(image_path, sample_name[0] +'.png')
            # print(final_path)
            vutils.save_image(perturbed_image, fp=final_path, nrow=1, padding=10)




def compare_two_net(args, val_loader, epoch, net: nn.Module, net2: nn.Module, clean_dir=True):
     # eval mode
    net.eval()
    net2.eval()
    dataset =os.path.basename(args.data_path)
    points =[]
    names =[]
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0 ,0 ,0 ,0), (0 ,0 ,0 ,0)
    rater_res = [(0 ,0 ,0 ,0) for _ in range(6)]
    hd =[]
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
            imgsw = pack['images'].to(dtype = torch.float32, device = GPUdevice)
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
                    pt = ptw[: ,: ,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[... ,buoy:buoy + evl_ch]
                masks = masksw[... ,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1 ,3 ,1 ,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size ,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size ,args.out_size))(masks)

                showp = pt
                points.append(pt.numpy()[0])
                names.append(*name)
                mask_type = torch.float32
                ind += 1
                b_size ,c ,w ,h = imgs.size()
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
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type ,device = GPUdevice)

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
                        coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
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
                    pred = F.interpolate(pred ,size=(masks.shape[2] ,masks.shape[3]))

                    "test net2"
                with torch.no_grad():
                    imge= net2.image_encoder(imgs)
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net2.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
                        se = net2.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'sam' or args.net == 'mobile_sam':
                        pred2, _ = net2.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net2.prompt_encoder.get_dense_pe(),
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
                        pred2, _ = net2.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net2.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )
                    pred2 = F.interpolate(pred2 ,size=(masks.shape[2] ,masks.shape[3]))


                    # print(pred.shape)
                    # temp_hd ,save_pred =calc_hf(pred.detach() ,masks)
                    temp_hd ,save_pred =calc_hf(pred.detach() ,pred2.detach())


                    # print(pack["image_meta_dict"]["filename_or_obj"])
                    hd.append(temp_hd)
                    # print(pred.shape,masks.shape,torch.max(pred),torch.max(masks),torch.min(masks))
                    tot += lossfunc(pred, pred2)
                    temp = eval_seg(pred, pred2, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()


    # return tot/ n_val , tuple([ a /n_val for a in mix_res]) ,sum(hd ) /len(val_loader)
    return tot/ n_val , tuple([ a /n_val for a in mix_res])




def heat_map(args, net, train_loader):
    # net.train()
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
    image_path = f"./dataset/TestDataset/heat_map/"
    Path(image_path).mkdir(parents=True, exist_ok=True)


    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(train_loader):
            imgsw = pack['images'].to(dtype=torch.float32, device=GPUdevice)
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
                    imgs = imgs.to(dtype=mask_type, device=GPUdevice).requires_grad_(True)
                # print(type(imgs))
                image_tensor = imgs.squeeze(0)  # Shape becomes [3, 1024, 1024]

                image_tensor = image_tensor.permute(1, 2, 0)
                torch.cuda.empty_cache()

                predictor = SamPredictor(net)

                predictor.set_image(np.array(image_tensor.cpu().detach().numpy()))

                # Define input points or bounding boxes for SAM
                input_points = np.array([[256, 256]])  # Example: Set a point in the middle of the image
                input_labels = np.array([1])  # Label 1 corresponds to a positive point for segmentation

                # Generate segmentation mask using SAM
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True  # Get multiple masks (you can adjust this based on your application)
                )

                # Select the mask with the highest score
                best_mask_index = np.argmax(scores)
                best_mask = masks[best_mask_index]
                cam_extractor = GradCAM(net.image_encoder, target_layer="blocks.11")

                # Generate Grad-CAM activation map for the given input image

                output = net.image_encoder(imgs)  # Forward pass through SAM encoder
                class_idx = int(best_mask_index)
                activation_map = cam_extractor(class_idx, output)
                print(activation_map)
                activation_map = activation_map.squeeze().cpu().numpy()
                heatmap = cv2.resize(activation_map, (masks.shape[2], masks.shape[3]))

                # Normalize the heatmap between 0 and 1
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                # Apply a colormap to the heatmap for visualization
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)



                # Overlay heatmap on original image
                overlay = overlay_mask(imgs, heatmap_colored, alpha=0.5)

                for na in name:
                    namecat = na.split('/')[-1].split('.')[0] + '+'
                final_path = os.path.join(image_path, namecat +'.png')
                print('final_path',final_path)
                vutils.save_image(overlay, fp=final_path, nrow=1, padding=10)

                # def backward_hook(module, grad_input, grad_output):
                #     global gradients # refers to the variable in the global scope
                #     print('Backward hook running...')
                #     gradients = grad_output
                #     # print(gradients)
                #     # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
                #     print(f'Gradients size: {gradients[0].size()}')
                #     # We need the 0 index because the tensor containing the gradients comes
                #     # inside a one element tuple.
                #
                # def forward_hook(module, args, output):
                #     global activations # refers to the variable in the global scope
                #     print('Forward hook running...')
                #     activations = output
                #     # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
                #     print(f'Activations size: {activations.size()}')
                # #
                # # backward_hook = net.mask_decoder.output_upscaling[3].register_full_backward_hook(backward_hook, prepend=False)
                # #
                # # forward_hook = net.mask_decoder.output_upscaling[3].register_forward_hook(forward_hook, prepend=False)
                #
                #
                # backward_hook = net.image_encoder.neck[3].register_full_backward_hook(backward_hook, prepend=False)
                # #
                # forward_hook = net.image_encoder.neck[3].register_forward_hook(forward_hook, prepend=False)
                #
                #
                # # backward_hook = net.mask_decoder.transformer.layers[0].cross_attn_token_to_image.register_full_backward_hook(backward_hook, prepend=False)
                # #
                # # forward_hook = net.mask_decoder.transformer.layers[0].cross_attn_token_to_image.register_forward_hook(forward_hook, prepend=False)
                #
                # imge= net.image_encoder(imgs)


                # if args.net == 'sam' or args.net == 'mobile_sam':
                #     se, de = net.prompt_encoder(
                #         points=pt,
                #         boxes=None,
                #         masks=None,
                #     )
                # elif args.net == "efficient_sam":
                #     coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
                #     se = net.prompt_encoder(
                #         coords=coords_torch,
                #         labels=labels_torch,
                #     )
                #
                # if args.net == 'sam' or args.net == 'mobile_sam':
                #     pred, _ = net.mask_decoder(
                #         image_embeddings=imge,
                #         image_pe=net.prompt_encoder.get_dense_pe(),
                #         sparse_prompt_embeddings=se,
                #         dense_prompt_embeddings=de,
                #         multimask_output=False,
                #     )
                #
                #
                # elif args.net == "efficient_sam":
                #     se = se.view(
                #         se.shape[0],
                #         1,
                #         se.shape[1],
                #         se.shape[2],
                #     )
                #     pred, _ = net.mask_decoder(
                #         image_embeddings=imge,
                #         image_pe=net.prompt_encoder.get_dense_pe(),
                #         sparse_prompt_embeddings=se,
                #         multimask_output=False,
                #     )
                #
                #
                # # Resize to the ordered output size
                # # pred = F.interpolate(pred, size=(args.out_size, args.out_size))
                # pred = F.interpolate(pred, size=(masks.shape[2], masks.shape[3]))
                # origin_pred = pred
                # # hd.append(calc_hf(pred,masks))
                # loss = lossfunc(pred, masks)
                #
                # loss.backward()
                #
                # weights = F.adaptive_avg_pool2d(gradients[0], 1)
                # gcam = torch.mul(activations, weights).sum(dim=1, keepdim=True)
                # gcam = F.relu(gcam)
                # gcam = F.interpolate(
                #     gcam, size=(1024, 1024)
                # )
                #
                # B, C, H, W = gcam.shape
                # gcam = gcam.view(B, -1)
                # gcam -= gcam.min(dim=1, keepdim=True)[0]
                # gcam /= gcam.max(dim=1, keepdim=True)[0]
                # gcam = gcam.view(B, C, H, W)
                # heatmap_ratio = 1
                # heatmap_colored = gcam
                #
                # overlayed_image = imgs * (1 - heatmap_ratio) + heatmap_colored * heatmap_ratio
                # # print(overlayed_image.size())
                #
                # # Convert overlayed_image to CPU and NumPy for plotting
                # overlay = overlayed_image.detach()
                # # print(overlay.size())
                # for na in name:
                #     namecat = na.split('/')[-1].split('.')[0] + '+'
                # final_path = os.path.join(image_path, namecat +'.png')
                # print('final_path',final_path)
                # vutils.save_image(overlay, fp=final_path, nrow=1, padding=10)








