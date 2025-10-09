import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch import nn, optim
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# 1. 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path).dropna()
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return (torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled),
            scaler_X, scaler_y)


x, y, scaler_x, scaler_y = load_data("train.csv")


# 2. 模型创建
def build_model():
    model = nn.Sequential(nn.Linear(1, 1))
    nn.init.normal_(model[0].weight, std=0.1)
    nn.init.normal_(model[0].bias, std=0.1)
    return model


# 3. 训练器类
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()
        self.history = {'loss': [], 'w': [], 'b': []}
        self.best_loss = float('inf')
        self.best_weights = None

    def train(self, x, y, epochs=100):
        for _ in range(epochs):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录训练过程
            self.history['loss'].append(loss.item())
            self.history['w'].append(self.model[0].weight.item())
            self.history['b'].append(self.model[0].bias.item())

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_weights = {
                    'w': self.model[0].weight.item(),
                    'b': self.model[0].bias.item()
                }
        return self.history, self.best_loss, self.best_weights


# 4. 优化器对比（更换为RMSprop、Rprop、SGD）
optimizers = {
    'RMSprop': lambda params: optim.RMSprop(params, lr=0.01),
    'Rprop': lambda params: optim.Rprop(params, lr=0.01),  # Rprop优化器
    'SGD': lambda params: optim.SGD(params, lr=0.01)
}

opt_results = {}
print("=== 优化器性能对比 ===")
for name, opt_fn in optimizers.items():
    model = build_model()
    trainer = Trainer(model, opt_fn(model.parameters()))
    hist, best_loss, weights = trainer.train(x, y, 100)
    opt_results[name] = {
        'history': hist,
        'best_loss': best_loss,
        'weights': weights
    }
    print(f"{name} 最佳损失: {best_loss:.6f}")


# 5. 优化器可视化
def plot_optimizers(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 损失曲线
    for name, data in results.items():
        axes[0].plot(data['history']['loss'], label=name)
    axes[0].set(title='损失曲线', xlabel='轮次', ylabel='损失值')
    axes[0].legend()

    # 权重变化
    for name, data in results.items():
        axes[1].plot(data['history']['w'], label=name)
    axes[1].set(title='权重w变化', xlabel='轮次', ylabel='w')

    # 偏置变化
    for name, data in results.items():
        axes[2].plot(data['history']['b'], label=name)
    axes[2].set(title='偏置b变化', xlabel='轮次', ylabel='b')

    # w-损失关系
    for name, data in results.items():
        axes[3].scatter(data['history']['w'], data['history']['loss'], label=name, s=3)
    axes[3].set(title='w与损失关系', xlabel='w', ylabel='损失值')

    # b-损失关系
    for name, data in results.items():
        axes[4].scatter(data['history']['b'], data['history']['loss'], label=name, s=3)
    axes[4].set(title='b与损失关系', xlabel='b', ylabel='损失值')

    # 参数轨迹
    for name, data in results.items():
        axes[5].plot(data['history']['w'], data['history']['b'], label=name, alpha=0.7)
    axes[5].set(title='参数空间轨迹', xlabel='w', ylabel='b')

    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_optimizers(opt_results)


# 6. 超参数实验（以SGD为例）
def run_param_test(param_type, values):
    results = {}
    print(f"\n=== {param_type}调节实验 ===")
    for val in values:
        model = build_model()
        if param_type == '学习率':
            opt = optim.SGD(model.parameters(), lr=val)
            trainer = Trainer(model, opt)
            hist, loss, _ = trainer.train(x, y, 100)
            key = f"LR={val}"
        else:
            opt = optim.SGD(model.parameters(), lr=0.01)
            trainer = Trainer(model, opt)
            hist, loss, _ = trainer.train(x, y, val)
            key = f"Epochs={val}"
        results[key] = {'history': hist, 'best_loss': loss}
        print(f"{key}: 最佳损失={loss:.6f}")
    return results


lr_results = run_param_test('学习率', [0.001, 0.01, 0.1])
epoch_results = run_param_test('训练轮数', [50, 100, 200])


# 7. 超参数可视化
def plot_params(lr_data, epoch_data):
    # 学习率图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in lr_data.items():
        ax1.plot(data['history']['loss'], label=name)
    ax1.set(title='学习率-损失曲线', xlabel='轮次', ylabel='损失值')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(lr_data.keys(), [d['best_loss'] for d in lr_data.values()])
    ax2.set(title='学习率-最佳损失', ylabel='损失值')
    ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # 轮数图表
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in epoch_data.items():
        ax3.plot(data['history']['loss'], label=name)
    ax3.set(title='轮数-损失曲线', xlabel='轮次', ylabel='损失值')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4.bar(epoch_data.keys(), [d['best_loss'] for d in epoch_data.values()])
    ax4.set(title='轮数-最佳损失', ylabel='损失值')
    ax4.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


plot_params(lr_results, epoch_results)

# 8. 保存最佳模型
all_results = {**opt_results, **lr_results, **epoch_results}
best_key = min(all_results.keys(), key=lambda k: all_results[k]['best_loss'])
best_data = all_results[best_key]

print(f"\n最佳模型: {best_key}，最佳损失: {best_data['best_loss']:.6f}")

os.makedirs("best_model", exist_ok=True)
torch.save({
    'best_loss': best_data['best_loss'],
    'x_mean': scaler_x.mean_[0], 'x_std': scaler_x.scale_[0],
    'y_mean': scaler_y.mean_[0], 'y_std': scaler_y.scale_[0],
    'best_weights': best_data.get('weights'),
    'model_name': best_key
}, "best_model/best_model.pt")