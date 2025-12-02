import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
import gzip
import struct

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 适配网络输入尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 统计均值和方差
])


def load_mnist_images(file_path):
    """读取 MNIST 图像文件（.gz 压缩格式）"""
    with gzip.open(file_path, 'rb') as f:
        # 解析文件头：magic number, num_images, rows, cols
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取图像数据并reshape
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(file_path):
    """读取 MNIST 标签文件（.gz 压缩格式）"""
    with gzip.open(file_path, 'rb') as f:
        # 解析文件头：magic number, num_labels
        magic, num = struct.unpack('>II', f.read(8))
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


class CustomMNISTDataset(torch.utils.data.Dataset):
    """自定义 Dataset 加载本地 MNIST .gz 文件"""

    def __init__(self, images_path, labels_path, transform=None):
        self.images = load_mnist_images(images_path)
        self.labels = load_mnist_labels(labels_path)
        self.transform = transform
        assert len(self.images) == len(self.labels), "图像和标签数量不匹配"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像（HWC 格式，灰度图）
        image = self.images[idx]
        # 转换为 PIL Image（torchvision.transforms 要求输入为 PIL 图像）
        image = transforms.ToPILImage()(image)
        label = self.labels[idx]

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据集路径配置
data_dir = r"D:\BaiduNetdiskDownload\MNIST"
train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

# 验证文件是否存在
file_paths = [train_images_path, train_labels_path, test_images_path, test_labels_path]
for path in file_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}\n请检查路径是否正确！")

# 加载自定义数据集
print("加载 MNIST 数据集...")
train_dataset = CustomMNISTDataset(
    images_path=train_images_path,
    labels_path=train_labels_path,
    transform=transform
)
test_dataset = CustomMNISTDataset(
    images_path=test_images_path,
    labels_path=test_labels_path,
    transform=transform
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # Windows 建议 num_workers=0
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# 网络模型定义（保持不变）
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        # 1x1+3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        # 1x1+5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        # 池化+1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        return outputs


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.a3 = Inception(64, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut连接（维度匹配）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 构建ResNet18
def ResNet18():
    return ResNet(ResBlock, [2, 2, 2, 2])


# 训练/测试函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} Train')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 实时更新进度条
        pbar.set_postfix({
            'Loss': f'{train_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    train_acc = 100. * correct / total
    train_avg_loss = train_loss / len(train_loader)
    return train_avg_loss, train_acc


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Test')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{test_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    test_acc = 100. * correct / total
    test_avg_loss = test_loss / len(test_loader)
    return test_avg_loss, test_acc


def train_model(model_name, model, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    times = []

    print(f"\n{model_name} 模型参数量: {count_params(model) / 1e6:.2f}M")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 打印epoch总结
        print(f'\nEpoch {epoch + 1} 总结 | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | '
              f'Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')

    total_time = time.time() - start_time
    print(f'\n{model_name} 总训练时间: {total_time:.2f}s')
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'epoch_times': times,
        'total_time': total_time
    }


# 参数量计算函数
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 初始化模型
    googlenet = GoogleNet().to(device)
    resnet18 = ResNet18().to(device)

    # 训练模型（10个epoch）
    print("=" * 50)
    print("Training GoogleNet...")
    googlenet_results = train_model("GoogleNet", googlenet, epochs=10)

    print("\n" + "=" * 50)
    print("Training ResNet18...")
    resnet_results = train_model("ResNet18", resnet18, epochs=10)

    # 可视化训练曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 4))

    # 准确率对比
    plt.subplot(1, 2, 1)
    plt.plot(googlenet_results['train_accs'], label='GoogleNet-训练准确率', marker='o', markersize=4)
    plt.plot(googlenet_results['test_accs'], label='GoogleNet-测试准确率', marker='s', markersize=4)
    plt.plot(resnet_results['train_accs'], label='ResNet18-训练准确率', marker='^', markersize=4)
    plt.plot(resnet_results['test_accs'], label='ResNet18-测试准确率', marker='d', markersize=4)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('准确率 (%)', fontsize=10)
    plt.title('训练/测试准确率对比', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(10))

    # 损失对比
    plt.subplot(1, 2, 2)
    plt.plot(googlenet_results['train_losses'], label='GoogleNet-训练损失', marker='o', markersize=4)
    plt.plot(googlenet_results['test_losses'], label='GoogleNet-测试损失', marker='s', markersize=4)
    plt.plot(resnet_results['train_losses'], label='ResNet18-训练损失', marker='^', markersize=4)
    plt.plot(resnet_results['test_losses'], label='ResNet18-测试损失', marker='d', markersize=4)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.title('训练/测试损失对比', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(10))

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 性能指标汇总
    print("\n" + "=" * 60)
    print("                      性能指标汇总")
    print("=" * 60)
    print(f"GoogleNet 最终测试准确率: {googlenet_results['test_accs'][-1]:.2f}%")
    print(f"ResNet18 最终测试准确率: {resnet_results['test_accs'][-1]:.2f}%")
    print(f"\nGoogleNet 总训练时间: {googlenet_results['total_time']:.2f}s")
    print(f"ResNet18 总训练时间: {resnet_results['total_time']:.2f}s")
    print(f"\nGoogleNet 参数量: {count_params(googlenet) / 1e6:.2f}M")
    print(f"ResNet18 参数量: {count_params(resnet18) / 1e6:.2f}M")