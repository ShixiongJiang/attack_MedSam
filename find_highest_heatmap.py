import numpy as np
import cv2
import torch
# 加载彩色热力图
image_path = "./106/heatmap-106.jpg"  # 替换为你的热力图路径
color_heatmap = cv2.imread(image_path)  # 加载彩色图像 (BGR 格式)
import os
# 计算每个像素的亮度值 (例如，RGB 平均值或加权)
# 使用加权公式计算亮度 (Y = 0.299*R + 0.587*G + 0.114*B)
brightness = 0.299 * color_heatmap[:, :, 2] + 0.587 * color_heatmap[:, :, 1] + 0.114 * color_heatmap[:, :, 0]

# 图像的高度和宽度
height, width = brightness.shape

# 窗口大小
window_size = 10

# 初始化变量
max_sum = -np.inf
best_start = (0, 0)

# 遍历所有可能的 10x10 区域
for y in range(height - window_size + 1):
    for x in range(width - window_size + 1):
        # 计算当前窗口的亮度总和
        window_sum = np.sum(brightness[y:y + window_size, x:x + window_size])
        # 更新最大值及其对应位置
        if window_sum > max_sum:
            max_sum = window_sum
            best_start = (x, y)

# 输出结果
print(f"最亮的 10x10 区域起始坐标: {best_start}, 总亮度值: {max_sum}")

# 在原图像上标记最亮区域
marked_image = color_heatmap.copy()
cv2.rectangle(marked_image, best_start, (best_start[0] + window_size, best_start[1] + window_size), (0, 0, 255), 2)

# 保存或显示结果
output_path = "./106/heatmap_revised.jpg"  # 替换为保存路径
cv2.imwrite(output_path, marked_image)
print(f"标记后的图像已保存到: {output_path}")
#
# # 显示结果 (可选)
#
# args=parse_args()
#
# # 检查设备
# GPUdevice = torch.device('cuda', args.gpu_device if args.gpu else 'cpu')
#
# net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice)
#
# # 加载预训练权重
# assert args.weights != 0, "Weight file not specified!"
# assert os.path.exists(args.weights), f"Weight file does not exist: {args.weights}"
# print(f'=> Resuming from {args.weights}')
#
# checkpoint_file = os.path.join(args.weights)
# loc = f'cuda:{args.gpu_device}'
# checkpoint = torch.load(checkpoint_file, map_location=loc)
# net.load_state_dict(checkpoint['state_dict'])
# # net.eval()
#
#
# # args, net, train_loader, color='black', log_dir="./heatmap_img/", attack_coords=None
# one_pixel_attack_specified(args,net,)
