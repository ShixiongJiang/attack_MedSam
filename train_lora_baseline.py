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
