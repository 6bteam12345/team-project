import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 1. 使用train.csv数据集
df = pd.read_csv(r"D:\神经网络\train.csv")
df_clean = df.dropna()
x_values = df_clean['x'].values
y_values = df_clean['y'].values

# 数据标准化
x_mean, x_std = x_values.mean(), x_values.std()
y_mean, y_std = y_values.mean(), y_values.std()
x_data = torch.FloatTensor((x_values - x_mean) / x_std).reshape(-1, 1)
y_data = torch.FloatTensor((y_values - y_mean) / y_std).reshape(-1, 1)

# 2. 初始化权重参数w和偏置参数b使其满足正态分布
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, std=0.1)  # 正态分布初始化w
        nn.init.normal_(self.linear.bias, std=0.1)    # 正态分布初始化b

    def forward(self, x):
        return self.linear(x)

# 3. 选择三种不同的优化器：Adagrad, Adam, SGD
optimizers_dict = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'Adagrad': torch.optim.Adagrad
}

# 存储训练结果
results = {}

# 训练函数
def train_with_optimizer(opt_name, opt_class, lr=0.01, epochs=100):
    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = opt_class(model.parameters(), lr=lr)

    history = {'loss': [], 'w': [], 'b': []}
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['w'].append(model.linear.weight.item())
        history['b'].append(model.linear.bias.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

    return history, best_loss, best_state

print("===三种优化器性能比较===")
for opt_name, opt_class in optimizers_dict.items():
    history, best_loss, best_state = train_with_optimizer(opt_name, opt_class)
    results[opt_name] = {
        'history': history,
        'best_loss': best_loss,
        'best_state': best_state
    }
    print(f"{opt_name}: 最佳损失={best_loss:.6f}")

# 4. 调节参数w和b的可视化
plt.figure(figsize=(15, 10))

# 损失曲线对比
plt.subplot(2, 3, 1)
for opt_name in optimizers_dict.keys():
    plt.plot(results[opt_name]['history']['loss'], label=opt_name)
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title('损失曲线对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 权重w变化过程
plt.subplot(2, 3, 2)
for opt_name in optimizers_dict.keys():
    plt.plot(results[opt_name]['history']['w'], label=opt_name)
plt.xlabel('训练轮数')
plt.ylabel('权重w')
plt.title('权重变化过程')
plt.legend()
plt.grid(True, alpha=0.3)

# 偏置b变化过程
plt.subplot(2, 3, 3)
for opt_name in optimizers_dict.keys():
    plt.plot(results[opt_name]['history']['b'], label=opt_name)
plt.xlabel('训练轮数')
plt.ylabel('偏置b')
plt.title('偏置变化过程')
plt.legend()
plt.grid(True, alpha=0.3)

# w与损失的关系
plt.subplot(2, 3, 4)
for opt_name in optimizers_dict.keys():
    history = results[opt_name]['history']
    plt.plot(history['w'], history['loss'], 'o-', label=opt_name, markersize=2)
plt.xlabel('权重w')
plt.ylabel('损失值')
plt.title('权重与损失的关系')
plt.legend()
plt.grid(True, alpha=0.3)

# b与损失的关系
plt.subplot(2, 3, 5)
for opt_name in optimizers_dict.keys():
    history = results[opt_name]['history']
    plt.plot(history['b'], history['loss'], 'o-', label=opt_name, markersize=2)
plt.xlabel('偏置b')
plt.ylabel('损失值')
plt.title('偏置与损失的关系')
plt.legend()
plt.grid(True, alpha=0.3)

# 参数空间轨迹
plt.subplot(2, 3, 6)
for opt_name in optimizers_dict.keys():
    history = results[opt_name]['history']
    plt.plot(history['w'], history['b'], 'o-', label=opt_name, markersize=2)
plt.xlabel('权重w')
plt.ylabel('偏置b')
plt.title('参数空间轨迹')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. 调节参数epoch和学习率η的可视化
print("\n===学习率调节实验===")
learning_rates = [0.001, 0.01, 0.1]
lr_results = {}
for lr in learning_rates:
    history, best_loss, _ = train_with_optimizer('SGD', torch.optim.SGD, lr=lr)
    lr_results[f'LR={lr}'] = {'history': history, 'best_loss': best_loss}
    print(f"学习率{lr}: 最佳损失={best_loss:.6f}")

print("\n===训练轮数调节实验===")
epochs_list = [50, 100, 200]
epoch_results = {}
for epochs in epochs_list:
    history, best_loss, _ = train_with_optimizer('SGD', torch.optim.SGD, epochs=epochs)
    epoch_results[f'Epochs={epochs}'] = {'history': history, 'best_loss': best_loss}
    print(f"训练轮数{epochs}: 最佳损失={best_loss:.6f}")

# 学习率调节可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for lr_name, result in lr_results.items():
    plt.plot(result['history']['loss'], label=lr_name)
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title('不同学习率的损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
lr_names = list(lr_results.keys())
lr_losses = [result['best_loss'] for result in lr_results.values()]
plt.bar(lr_names, lr_losses, alpha=0.7)
plt.ylabel('最佳损失值')
plt.title('学习率与最佳损失的关系')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 训练轮数调节可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for epoch_name, result in epoch_results.items():
    plt.plot(result['history']['loss'], label=epoch_name)
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title('不同训练轮数的损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
epoch_names = list(epoch_results.keys())
epoch_losses = [result['best_loss'] for result in epoch_results.values()]
plt.bar(epoch_names, epoch_losses, alpha=0.7)
plt.ylabel('最佳损失值')
plt.title('训练轮数与最佳损失的关系')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 将训练性能最好的代码保存下来
print("\n===保存最佳模型===")
# 找到全局最佳模型
all_results = {** results, **lr_results,** epoch_results}
best_model_name = min(all_results.keys(), key=lambda x: all_results[x]['best_loss'])
best_result = all_results[best_model_name]

print(f"全局最佳模型: {best_model_name}")
print(f"最佳损失值: {best_result['best_loss']:.6f}")

# 创建最佳模型并加载参数
best_model = LinearModel()
if 'best_state' in best_result:
    best_model.load_state_dict(best_result['best_state'])

# 保存模型
os.makedirs("best_model", exist_ok=True)
torch.save({
    'model_state_dict': best_model.state_dict(),
    'best_loss': best_result['best_loss'],
    'x_mean': x_mean, 'x_std': x_std,
    'y_mean': y_mean, 'y_std': y_std,
    'model_name': best_model_name
}, "best_model/best_model.pth")

print("最佳模型已保存到: best_model/best_model.pth")
