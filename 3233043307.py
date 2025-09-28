import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

# 1. 加载并预处理数据
df = pd.read_csv(r"D:\神经网络\train.csv")

# 查看数据基本信息
print("数据集基本信息：")
print(df.info())

# 处理缺失值：使用每列的平均值填充
df = df.fillna(df.mean())

# 假设最后一列是目标变量，前面的是特征
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 2. 定义线性模型
class LinearModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)  # 输入维度，输出维度

    def forward(self, x):
        return self.linear(x)


# 初始化模型
input_size = X_train.shape[1]
model = LinearModel(input_size)

# 3. 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
epochs = 1000
losses = []
weights = []

for epoch in range(epochs):
    # 前向传播
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # 记录损失和权重
    losses.append(loss.item())
    # 收集所有权重（包括偏置）
    current_weights = []
    for param in model.parameters():
        current_weights.extend(param.data.numpy().flatten())
    weights.append(current_weights)

    # 反向传播和优化
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 结果可视化
plt.figure(figsize=(15, 10))

# 绘制损失变化
plt.subplot(2, 1, 1)
plt.plot(range(epochs), losses)
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# 绘制权重变化
plt.subplot(2, 1, 2)
weights = np.array(weights)
for i in range(weights.shape[1]):
    plt.plot(range(epochs), weights[:, i], label=f'Weight {i + 1}')
plt.title('Weights vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. 测试模型
with torch.no_grad():  # 测试时不需要计算梯度
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# 打印最终权重
print("\n最终权重:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy()}")
