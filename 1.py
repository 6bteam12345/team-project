import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载和预处理数据
try:
    # 尝试读取本地train.csv文件
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    # 如果本地文件不存在，使用示例数据并保存为train.csv
    print("本地train.csv文件未找到，使用示例数据...")
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
    df = pd.read_csv(url)
    df.to_csv('train.csv', index=False)

# 显示数据集信息
print("数据集信息：")
print(df.info())
print("\n缺失值统计：")
print(df.isnull().sum())

# 处理缺失值：使用每列的平均值填充
df = df.fillna(df.mean())

# 假设最后一列是目标变量，其余是特征
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# 将类别特征转换为数值特征（如果有）
for i in range(X.shape[1]):
    if np.issubdtype(X[:, i].dtype, np.object_):
        X[:, i] = pd.factorize(X[:, i])[0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 转换为PyTorch张量
X_train = torch.tensor(X_train_scaled, dtype=torch.float32, requires_grad=False)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32, requires_grad=False)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32, requires_grad=False)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32, requires_grad=False)

# 2. 初始化模型参数（权重和偏置）
input_size = X_train.shape[1]
w = torch.randn(input_size, 1, dtype=torch.float32, requires_grad=True)  # 权重
b = torch.randn(1, dtype=torch.float32, requires_grad=True)  # 偏置


# 3. 定义模型和训练参数
def model(x):
    return torch.matmul(x, w) + b  # 线性模型：y = x*w + b


learning_rate = 0.01
epochs = 1000

# 记录训练过程中的参数和损失
loss_history = []
weights_history = []
bias_history = []

# 4. 训练模型
for epoch in range(epochs):
    # 前向传播：计算预测值
    y_pred = model(X_train)

    # 计算损失（均方误差）
    loss = torch.mean((y_pred - y_train) ** 2)
    loss_history.append(loss.item())

    # 记录当前权重和偏置
    weights_history.append(w.detach().numpy().copy())
    bias_history.append(b.item())

    # 反向传播：计算梯度
    loss.backward()

    # 查看梯度（使用grad相关属性）
    if epoch % 100 == 0:
        print(f"\nEpoch {epoch}")
        print(f"Loss: {loss.item():.6f}")
        print(f"部分权重梯度: {w.grad[:2].numpy().flatten()}")
        print(f"偏置梯度: {b.grad.item()}")

    # 更新参数（不跟踪梯度）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

# 5. 可视化结果

# 创建画布
plt.figure(figsize=(15, 10))

# 绘制损失变化曲线
plt.subplot(2, 2, 1)
plt.plot(range(epochs), loss_history, 'b-')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# 绘制权重变化曲线（绘制前3个权重）
plt.subplot(2, 2, 2)
num_weights = min(3, input_size)
for i in range(num_weights):
    weights = [wh[i, 0] for wh in weights_history]
    plt.plot(range(epochs), weights, label=f'w_{i + 1}')
plt.title('Weights vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)

# 绘制偏置变化曲线
plt.subplot(2, 2, 3)
plt.plot(range(epochs), bias_history, 'r-')
plt.title('Bias vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Bias Value')
plt.grid(True)

# 绘制预测值与真实值对比
plt.subplot(2, 2, 4)
with torch.no_grad():
    y_pred_test = model(X_test)

plt.scatter(y_test.numpy(), y_pred_test.numpy(), alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印最终参数
print("\n最终模型参数:")
print(f"权重 w: {w.detach().numpy().flatten()}")
print(f"偏置 b: {b.item()}")
print(f"最终损失: {loss_history[-1]:.6f}")
