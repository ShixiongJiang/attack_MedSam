import argparse
import os

# from your_model_library import get_network  # Replace with your SAM model import
# from your_visualization_library import visualize_result  # Replace with your visualization function
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils import get_network
from cfg_reverse_adaptation import parse_args


args=parse_args()

# 检查设备
GPUdevice = torch.device('cuda', args.gpu_device if args.gpu else 'cpu')

# 获取网络模型
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice)

# 加载预训练权重
assert args.weights != 0, "Weight file not specified!"
assert os.path.exists(args.weights), f"Weight file does not exist: {args.weights}"
print(f'=> Resuming from {args.weights}')

checkpoint_file = os.path.join(args.weights)
loc = f'cuda:{args.gpu_device}'
checkpoint = torch.load(checkpoint_file, map_location=loc)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
print(f"Model loaded successfully from {args.weights}")

# 加载测试图像
image_path = "106/original-106.jpg"  # Replace with your image path
assert os.path.exists(image_path), f"Image does not exist: {image_path}"

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(GPUdevice)

# 前向推理
with torch.no_grad():
    output = net(input_tensor)

# 可视化结果
print(output.shape,'this is output shape')
# 如果是概率值，先将其转换为二值掩码
output_np = output.squeeze().cpu().numpy()  # [1024, 1024]

# 二值化并扩展到 0-255 范围
output_mask = (output_np > 0.5).astype(np.uint8) * 255

# 转为 PIL 图像
result_image = Image.fromarray(output_mask)

# 保存到指定路径
output_path = "./106/result.png"  # 替换为你的保存路径
result_image.save(output_path)
print(f"Result saved as image at {output_path}")