from einops import rearrange

def evolutionary_algorithm(args, net, train_loader, heatmap_img_path, color='black'):
    n_val = len(train_loader)

    # ... existing code until patch placement section ...
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(train_loader):
            # remove unnecessary continue
            imgsw = pack['images'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            names_batch = pack['image_meta_dict']['filename_or_obj']
            for name in names_batch:
                namecat = os.path.splitext(os.path.basename(name))[0] + '+'

            buoy = 0
            evl_ch = int(args.evl_chunk) if args.evl_chunk else int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                pt = ptw[:, :, buoy: buoy + evl_ch] if args.thd else ptw
                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    resize_transform = torchvision.transforms.Resize((args.image_size, args.image_size))
                    imgs = resize_transform(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
            _imgs = imgsw.clone()
            mask_type = torch.float32

            # Evolutionary algorithm parameters
            patch_size = 10
            color_value = 255 if color == 'white' else 0
            population_size = 30
            generations = 50
            mutation_rate = 0.1
            threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

            tournament_size = 3
            
            def create_individual():
                """Create a random patch position"""
                return [
                    random.randint(patch_size - 1, args.image_size - patch_size),
                    random.randint(patch_size - 1, args.image_size - patch_size)
                ]

            def fitness(position):
                """Evaluate fitness of a patch position (lower IoU is better)"""
                imgs = _imgs.clone()

                i, j = position
                imgs[0, :, i - patch_size + 1:i + 1,
                        j - patch_size + 1:j + 1] = color_value
                imgs = imgs.to(dtype=mask_type, device=GPUdevice)
                
                
                with torch.no_grad():
                    imge = net.image_encoder(imgs)
                    if args.net in ['sam', 'mobile_sam']:
                        se, de = net.prompt_encoder(points=pt, boxes=None, masks=None)
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                        )
                

                    pred = F.interpolate(pred, size=(masks.shape[2], masks.shape[3]))
                    eiou, _ = eval_seg(pred, masks, threshold)
                    return -eiou  # Negative because we want to minimize IoU

            def tournament_select(population, fitnesses):
                """Tournament selection"""
                tournament = random.sample(list(range(len(population))), tournament_size)
                winner = tournament[0]
                for idx in tournament[1:]:
                    if fitnesses[idx] > fitnesses[winner]:
                        winner = idx
                return population[winner]

            def mutate(position):
                """Mutate position with gaussian noise"""
                new_pos = position.copy()
                if random.random() < mutation_rate:
                    new_pos[0] += int(random.gauss(0, args.image_size/10))
                    new_pos[1] += int(random.gauss(0, args.image_size/10))
                    new_pos[0] = max(patch_size - 1, min(new_pos[0], args.image_size - patch_size))
                    new_pos[1] = max(patch_size - 1, min(new_pos[1], args.image_size - patch_size))
                return new_pos

            # Initialize population
            population = [create_individual() for _ in range(population_size)]
            best_position = None
            best_fitness = float('-inf')
            eiou_list = []
            pos_list = []

            # Evolution loop
            for gen in range(generations):
                # Evaluate fitness for all individuals
                fitnesses = [fitness(pos) for pos in population]
                
                # Track best solution
                gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
                if fitnesses[gen_best_idx] > best_fitness:
                    best_fitness = fitnesses[gen_best_idx]
                    best_position = population[gen_best_idx]
                    
                # Create new population
                new_population = []
                for _ in range(population_size):
                    parent = tournament_select(population, fitnesses)
                    child = mutate(parent)
                    new_population.append(child)
                
                population = new_population
                
                # Save position and metrics for visualization
                pos_list.append(best_position)
                eiou_list.append(-best_fitness)  # Convert back to IoU

            eiou_list = np.array(eiou_list)
            print(eiou_list)

    # ... rest of the visualization code ...



# train_reverse_adaptation.py
# !/usr/bin/env	python3

import re

from tensorboardX import SummaryWriter
# from dataset import *
from torch.utils.data import DataLoader, random_split

import cfg_reverse_adaptation
import function_r as function
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *


def remove_dumplicate_image(polyp_train_dataset=None, heatmap_img_path=None):
    # 假设 heatmap_img 路径
    # heatmap_img_path = 'heatmap_img_CVC300'
    if not os.path.exists(heatmap_img_path):
        os.mkdir(heatmap_img_path)
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
        print('image name', image_name)
        if image_name not in processed_images:
            filtered_data.append(data)
        else:
            print('dumplicate image', data['image_meta_dict']['filename_or_obj'])
    # 更新数据集
    polyp_train_dataset = filtered_data
    return polyp_train_dataset


# 获取参数
args = cfg_reverse_adaptation.parse_args()

# 确保 gpu_device 为 0
args.gpu_device = 0
# 打印当前进程的 CUDA_VISIBLE_DEVICES
print(f"Process {args.process_idx} sees CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
# 设置设备
GPUdevice = torch.device('cuda', args.gpu_device)
print(f'Using GPU: {args.gpu_device} (logical GPU in this process) for process index: {args.process_idx}')

# 初始化网络并加载模型
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

# 加载预训练模型
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = f'cuda:{args.gpu_device}'
checkpoint = torch.load(checkpoint_file, map_location=loc)

start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']
state_dict = checkpoint['state_dict']

# 根据是否分布式训练设置模型参数
if args.distributed != 'none':
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  # 保留 `module.` 前缀
        new_state_dict[name] = v
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

# 初始化优化器和学习率调度器
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

# 设置日志
args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

# 数据增强设置
transform_train = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size, args.out_size)),
    transforms.ToTensor(),
])

# 加载并划分数据集
heatmap_img_path = 'heatmap_img_CVC300'
polyp_train_dataset = Polyp(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg,
                            mode='Training')
# polyp_train_dataset = remove_dumplicate_image(heatmap_img_path=heatmap_img_path, polyp_train_dataset=polyp_train_dataset)

# # 根据进程数分割数据集
# N = args.num_processes
# dataset_size = len(polyp_train_dataset)
# subset_sizes = [dataset_size // N] * N
# subset_sizes[-1] += dataset_size - sum(subset_sizes)
# subsets = random_split(polyp_train_dataset, subset_sizes)
# subset = subsets[args.process_idx]

# # 创建 DataLoader
nice_train_loader = DataLoader(polyp_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)

# 为每个进程创建一个单独的日志文件
try:
    os.mkdir('./attack_log')
except:
    pass
log_file_path = f"./attack_log/process_{args.process_idx}_data_log.txt"
with open(log_file_path, 'w') as log_file:
    # 遍历 train_loader 并写入图片名称
    for batch in nice_train_loader:
        # 假设图像路径在 batch 数据中 (根据具体数据集结构调整获取方式)
        image_paths = batch['image_meta_dict']['filename_or_obj']  # 假设 batch 中包含 'image_paths' 字段
        for path in image_paths:
            log_file.write(f"{path}\n")

# 设置每个进程的 TensorBoard 日志目录
writer = SummaryWriter(log_dir=f"./logs/process_{args.process_idx}")

# 启动攻击
evolutionary_algorithm(args, net, nice_train_loader, color='black', heatmap_img_path=heatmap_img_path)

# 关闭日志
writer.close()
