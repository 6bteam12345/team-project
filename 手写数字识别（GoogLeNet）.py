import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import numpy as np

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 全局参数设置
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0003
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")
print(f"使用计算设备: {DEVICE}")

# 数据预处理（MNIST适配GoogLeNet输入）
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # GoogLeNet标准输入尺寸224×224
    transforms.Grayscale(num_output_channels=3),  # 单通道转3通道（模拟RGB）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计归一化
                         std=[0.229, 0.224, 0.225])
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ===================== GoogLeNet核心模块：Inception模块 =====================
class Inception(nn.Module):
    """Inception模块：多尺度卷积核并行提取特征，拼接融合"""

    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool):
        super(Inception, self).__init__()
        # 分支1：1×1卷积（降维+特征提取）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 分支2：1×1降维 + 3×3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, stride=1, padding=1)  # padding=1保持尺寸
        )

        # 分支3：1×1降维 + 5×5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, stride=1, padding=2)  # padding=2保持尺寸
        )

        # 分支4：3×3池化 + 1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 池化后尺寸不变
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 4个分支并行计算，按通道维度拼接（特征融合）
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # dim=1：通道维度拼接


# ===================== GoogLeNet模型定义（适配MNIST 10分类）=====================
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        # 初始卷积层（预处理特征）
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 3×224×224 → 64×112×112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64×112×112 → 64×56×56
            nn.LocalResponseNorm(64),  # LRN层（GoogLeNet原设计，增强泛化）

            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # 64×56×56 → 192×56×56
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 192×56×56 → 192×28×28
        )

        # Inception模块组（5个Inception，按原论文参数配置）
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  # 192→256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)  # 256→480
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)  # 480×28×28 → 480×14×14

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)  # 480→512
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)  # 512→512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)  # 512→512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)  # 512→528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)  # 528→832
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)  # 832×14×14 → 832×7×7

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)  # 832→832
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)  # 832→1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化→1024×1×1

        # 分类头（Dropout+全连接）
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),  # GoogLeNet原论文Dropout率0.4
            nn.Linear(1024, num_classes)  # 1024→10（手写数字分类数）
        )

    def forward(self, x):
        # 预处理特征
        x = self.pre_layers(x)
        # Inception模块特征提取
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        # 全局平均池化（简化特征维度）
        x = self.avgpool(x)
        # 展平适配全连接层
        x = x.view(x.size(0), -1)
        # 分类输出
        x = self.classifier(x)
        return x


# ===================== 训练与测试函数 =====================
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播+参数更新
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印训练日志
        if batch_idx % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress = batch_idx * len(data)
            total = len(train_loader.dataset)
            print(f"Epoch {epoch} [{progress}/{total}] 损失: {avg_loss:.4f} 耗时: {time.time() - start_time:.2f}s")

    return total_loss / len(train_loader)


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 计算测试损失
            test_loss += criterion(output, target).item()
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 统计结果
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f"\n测试集：平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} "
          f"({test_acc:.2f}%)\n")
    return test_loss, test_acc


# ===================== 主函数 =====================
def main():
    # 初始化模型、损失函数、优化器
    model = GoogLeNet(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print("GoogLeNet模型结构（适配MNIST）:")
    print(model)

    # 训练记录
    train_losses = []
    test_losses = []
    test_accs = []

    # 开始训练
    print(f"\n开始训练（共{EPOCHS}轮）...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test_model(model, test_loader, criterion)

        # 记录结果
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # 可视化训练结果
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('GoogLeNet训练/测试损失变化')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='测试准确率', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.title('GoogLeNet测试准确率变化')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)
    plt.ylim(97, 100)  # 聚焦高准确率区间

    plt.tight_layout()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'googlenet_mnist.pth')
    print("\n模型已保存为 'googlenet_mnist.pth'")


if __name__ == '__main__':
    main()