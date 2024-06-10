import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

# 使用预训练模型（例如 ResNet）
model = models.resnet18(pretrained=True)
model.eval()

# 选择一个目标测试点
x_test, y_test = next(iter(train_loader))
x_test = x_test.cuda()
y_test = y_test.cuda()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 生成扰动
delta = torch.zeros_like(x_test, requires_grad=True)

# 定义优化器
optimizer = optim.SGD([delta], lr=0.01)

# 迭代优化扰动
num_iterations = 100
for _ in range(num_iterations):
    for x_train, y_train in train_loader:
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        
        optimizer.zero_grad()
        
        # 计算扰动后的训练数据损失
        outputs_train = model(x_train + delta)
        loss_train = loss_fn(outputs_train, y_train)
        
        # 计算目标测试点的损失
        outputs_test = model(x_test)
        loss_test = loss_fn(outputs_test, y_test)
        
        # 计算梯度
        grad_train = torch.autograd.grad(loss_train, model.parameters(), create_graph=True)
        grad_test = torch.autograd.grad(loss_test, model.parameters())
        
        # 计算目标函数
        objective = -sum((g_t * g).sum() for g_t, g in zip(grad_test, grad_train))
        
        # 计算目标函数对扰动的梯度并更新扰动
        objective.backward()
        optimizer.step()

# 将扰动应用于整个训练集并保存扰动后的图像
for i, (x_train, y_train) in enumerate(train_loader):
    x_train = x_train.cuda()
    perturbed_image = x_train + delta
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 确保像素值在有效范围内
    
    # 保存扰动后的图像
    perturbed_image_np = perturbed_image.cpu().detach().numpy()
    # 这里可以使用图像保存库（例如 PIL 或 OpenCV）来保存图像

    if i == 0:
        break  # 为了示例，只处理一个批次

print("扰动优化完成并应用于图像数据集。")
