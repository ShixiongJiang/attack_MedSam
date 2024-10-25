# # train_reverse_adaptation.py
# # !/usr/bin/env	python3
#
# """ train network using pytorch
#     Junde Wu
# """
#
# from pathlib import Path
#
# from tensorboardX import SummaryWriter
# # from dataset import *
# from torch.utils.data import DataLoader
#
# import cfg_reverse_adaptation
# import function_r as function
# from conf import settings
# # from models.discriminatorlayer import discriminator
# from dataset import *
# from utils import *
#
# args = cfg_reverse_adaptation.parse_args()
#
# GPUdevice = torch.device('cuda', args.gpu_device)
#
# net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
#
# '''load pretrained model'''
# assert args.weights != 0
# print(f'=> resuming from {args.weights}')
# assert os.path.exists(args.weights)
# checkpoint_file = os.path.join(args.weights)
# assert os.path.exists(checkpoint_file)
# loc = 'cuda:{}'.format(args.gpu_device)
# checkpoint = torch.load(checkpoint_file, map_location=loc)
# start_epoch = checkpoint['epoch']
# best_tol = checkpoint['best_tol']
#
# state_dict = checkpoint['state_dict']
# if args.distributed != 'none':
#     from collections import OrderedDict
#
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         # name = k[7:] # remove `module.`
#         name = 'module.' + k
#         new_state_dict[name] = v
#     # load params
# else:
#     new_state_dict = state_dict
#
# net.load_state_dict(new_state_dict)
#
# optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay
#
# # for n, p in net.named_parameters():
# #         if 'lora_' in n:
# #             print(n)
# #             print(p)
# # print(net)
# args.path_helper = set_log_dir('logs', args.exp_name)
# logger = create_logger(args.path_helper['log_path'])
# logger.info(args)
#
# '''segmentation data'''
# transform_train = transforms.Compose([
#     transforms.Resize((args.image_size, args.image_size)),
#     transforms.ToTensor(),
# ])
#
# transform_train_seg = transforms.Compose([
#     transforms.Resize((args.out_size, args.out_size)),
#     transforms.ToTensor(),
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize((args.image_size, args.image_size)),
#     transforms.ToTensor(),
# ])
#
# transform_test_seg = transforms.Compose([
#     transforms.Resize((args.out_size, args.out_size)),
#     transforms.ToTensor(),
# ])
#
# print('if poison ', args.poison)
# '''polyp data using the original dataset to generate poison sample'''
# polyp_train_dataset = Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
#                             mode='Training')
# polyp_test_dataset = Polyp(args, args.data_path, transform=transform_test, transform_msk=transform_test_seg,
#                            mode='Test')
#
# nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
# nice_test_loader = DataLoader(polyp_test_dataset, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)
# # '''checkpoint path and tensorboard'''
# # iter_per_epoch = len(Glaucoma_training_loader)
# checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
#                                f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
# # use tensorboard
# if not os.path.exists(settings.LOG_DIR):
#     os.mkdir(settings.LOG_DIR)
#
# Path(os.path.join(
#     settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")).mkdir(parents=True,
#                                                                                                     exist_ok=True)
#
# writer = SummaryWriter(log_dir=os.path.join(
#     settings.LOG_DIR, args.net, f"train_exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"))
# # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# # writer.add_graph(net, Variable(input_tensor, requires_grad=True))
#
# # create checkpoint folder to save model
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)
# checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
#
# '''begin training'''
# best_acc = 0.0
# best_tol = 1e4
# total_epoch = 1
#
# # for epoch in range(settings.EPOCH):
# for epoch in range(total_epoch):
#     # function.one_pixel_attack(args, net, nice_train_loader, color='black')
#     function.one_pixel_attack(args, net, nice_train_loader, color='white')
#
# writer.close()
from pathlib import Path
import re
from tensorboardX import SummaryWriter
# from dataset import *
from torch.utils.data import DataLoader, random_split
import cfg_reverse_adaptation
import function_r as function
from conf import settings
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
from cfg_reverse_adaptation import parse_args

def remove_dumplicate_image(polyp_train_dataset):
    # 假设 heatmap_img 路径
    heatmap_img_path = 'heatmap_img'

    # 列出 heatmap_img 下所有的文件并提取已处理的图片名称
    pattern = re.compile(r'orig_(\d+)\+\.png')

    # 列出 heatmap_img 下所有的文件并提取已处理图片的数字名称
    processed_images = set()
    for file in os.listdir(heatmap_img_path):
        match = pattern.search(file)
        if match:
            processed_images.add(match.group(1))
    # 创建一个新的数据集列表，存储未处理过的图片
    filtered_data = []
    print(processed_images)
    # 遍历原始数据集，检查每个图片是否在已处理的图片列表中
    for data in polyp_train_dataset:
        image_name = os.path.basename(data['image_meta_dict']['filename_or_obj'])  # 假设图片路径在数据字典中的键是 'image_path'
        print('image name',image_name)
        if image_name not in processed_images:
            filtered_data.append(data)
        else:
            print('dumplicate image',data['image_meta_dict']['filename_or_obj'])
    # 更新数据集
    polyp_train_dataset = filtered_data
    return polyp_train_dataset


args = cfg_reverse_adaptation.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)

checkpoint = torch.load(checkpoint_file, map_location=loc)

start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay

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

print('if poison ', args.poison)
'''polyp data using the original dataset to generate poison sample'''
polyp_train_dataset = Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
                            mode='Training')

polyp_train_dataset=remove_dumplicate_image(polyp_train_dataset)

# according to process split dataset to N parts
# each process indexed by process idx
N = args.num_processes

dataset_size = len(polyp_train_dataset)
subset_sizes = [dataset_size // N] * N
subset_sizes[-1] += dataset_size - sum(subset_sizes)

subsets = random_split(polyp_train_dataset, subset_sizes)

# 根据进程索引选择子数据集
subset = subsets[args.process_idx]

# 创建 DataLoader
nice_train_loader = DataLoader(subset, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)

# 设置日志目录
log_dir = f"./logs/process_{args.process_idx}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
# ... 其他设置 ...

# 开始攻击
function.one_pixel_attack(args, net, nice_train_loader, color='white')
writer.close()
