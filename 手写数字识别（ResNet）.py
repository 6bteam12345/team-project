import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 全局参数（针对MNIST优化，提升速度）
BATCH_SIZE = 256  # 增大批次提升GPU利用率
EPOCHS = 10
LEARNING_RATE = 0.001
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")
print(f"使用计算设备: {DEVICE}")

# 数据预处理（轻量化适配，避免冗余计算）
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 缩小输入尺寸（比224小得多）
    transforms.Grayscale(num_output_channels=1),  # 保留单通道（MNIST原生格式）
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 用MNIST自身的均值/方差
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


# ===================== ResNet核心模块：残差块（BasicBlock）=====================
class BasicBlock(nn.Module):
    """简化的残差块：2个卷积层 + 残差连接（解决深层网络梯度消失问题）"""
    expansion = 1  # 输出通道数与输入一致（简化版）

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # BatchNorm加速训练
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        #  shortcut连接：若输入输出尺寸/通道不一致，用1×1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接：主路径 + shortcut路径
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ===================== ResNet模型定义（简化版，适配MNIST）=====================
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16  # 初始通道数（轻量化设计）

        # 初始卷积层（32×32×1 → 32×32×16）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 残差层（4个残差块，分3组）
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # 32×32×16
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # 16×16×32（下采样）
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 8×8×64（下采样）

        # 分类头（全局平均池化 + 全连接）
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """构建残差层：包含多个残差块"""
        strides = [stride] + [1] * (num_blocks - 1)  # 仅第一个块下采样
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始特征提取
        out = F.relu(self.bn1(self.conv1(x)))
        # 残差块特征提取
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 全局平均池化（8×8×64 → 1×1×64）
        out = F.avg_pool2d(out, 8)
        # 展平分类
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 定义简化版ResNet（8层：2+2+2残差块，比ResNet18轻量）
def ResNet8():
    return ResNet(BasicBlock, [2, 2, 2])


# ===================== 训练与测试函数 =====================
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 每100批次打印一次日志
        if batch_idx % 100 == 0:
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
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f"\n测试集：平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n")
    return test_loss, test_acc


# ===================== 主函数 =====================
def main():
    # 初始化模型（轻量版ResNet8）、损失函数、优化器
    model = ResNet8().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("简化版ResNet模型结构（适配MNIST）:")
    print(model)

    # 训练记录
    train_losses = []
    test_losses = []
    test_accs = []

    # 开始训练（10轮）
    print(f"\n开始训练（共{EPOCHS}轮）...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test_model(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # 可视化结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('ResNet训练/测试损失变化')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='测试准确率', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.title('ResNet测试准确率变化')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)
    plt.ylim(97, 100)

    plt.tight_layout()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'resnet_mnist.pth')
    print("\n模型已保存为 'resnet_mnist.pth'")


if __name__ == '__main__':
    main()