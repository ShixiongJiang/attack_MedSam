# train_reverse_adaptation.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader
from pathlib import Path

import cfg_reverse_adaptation
import function_r as function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

from monai.metrics import  compute_hausdorff_distance ,DiceMetric
from monai.losses import  DiceCELoss
from einops import rearrange

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

def validation_sam(args, train_loader, epoch, net: nn.Module, clean_dir=True):
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
            intermediate_activations = {}

            # Function to capture the intermediate output
            def capture_activations(layer_name):
                def hook(module, input, output):
                    intermediate_activations[layer_name] = output
                return hook
             # Register forward hooks on the layers where you want to capture outputs
            net.image_encoder.blocks[0].attn.qkv.register_forward_hook(capture_activations('blocks_attn_loraB'))
            imge= net.image_encoder(imgs)


            with torch.no_grad():

                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
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

    return loss

args = cfg_reverse_adaptation.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

# for n, p in net.named_parameters():
#         if 'lora_' in n:
#             print(n)
#             print(p)
# print(net)
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
    tol, eiou, edice = validation_sam(args, nice_test_loader, epoch, net, writer)



