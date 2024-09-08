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

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.train()
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
                    intermediate_activations = {}

                    # Function to capture the intermediate output
                    def capture_activations(layer_name):
                        def hook(module, input, output):
                            intermediate_activations[layer_name] = output
                        return hook
                     # Register forward hooks on the layers where you want to capture outputs
                    net.image_encoder.blocks[0].attn.qkv.register_forward_hook(capture_activations('blocks_attn_loraB'))
                    imge= net.image_encoder(imgs)
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
                    pred = F.interpolate(pred ,size=(masks.shape[2] ,masks.shape[3]))


                    # print(intermediate_activations)
                    print(net.image_encoder.blocks[0].attn.qkv.lora_output.grad)


            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([ a /n_val for a in mix_res]) ,sum(hd ) /len(val_loader)

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



