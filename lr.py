import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
try:
    # 尝试读取train.csv文件
    df = pd.read_csv(r'D:\神经网络\train.csv')
    # 假设数据集中有'x'和'y'两列
    x_data = df['x'].values
    y_data = df['y'].values
    print(f"成功读取数据，共{len(x_data)}个样本")
except FileNotFoundError:
    # 如果没有找到文件，使用示例数据
    print("未找到train.csv文件，使用示例数据进行演示")
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([2.5, 4.5, 6.5, 8.5, 10.5])  # 实际关系为y=2x+0.5

# 2. 定义模型和损失函数（包含偏置项b）
def forward(x, w, b):
    """正向传播：y_pred = w*x + b"""
    return w * x + b

def loss(x, y, w, b):
    """计算损失：MSE损失"""
    y_pred = forward(x, w, b)
    return (y_pred - y) **2

# 3. 计算不同w和b组合下的损失值
# 定义w和b的取值范围
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 4.1, 0.1)

# 存储w、b和对应的损失值
w_list = []
b_list = []
mse_list = []

# 计算所有组合的损失
for w in w_range:
    for b in b_range:
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, w, b)
        mse = l_sum / len(x_data)  # 计算平均损失
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 4. 可视化w与loss的关系（固定b为最优值附近）
# 找到损失最小时的b值索引
min_loss_idx = np.argmin(mse_list)
optimal_b = b_list[min_loss_idx]

# 提取该b值下的w和对应的loss
fixed_b_w = []
fixed_b_loss = []
for w, b, loss_val in zip(w_list, b_list, mse_list):
    if abs(b - optimal_b) < 0.01:  # 找到最优b附近的值
        fixed_b_w.append(w)
        fixed_b_loss.append(loss_val)

plt.figure(figsize=(12, 5))

# 绘制w与loss的关系图
plt.subplot(1, 2, 1)
plt.plot(fixed_b_w, fixed_b_loss, 'b-')
plt.axvline(x=w_list[min_loss_idx], color='r', linestyle='--', label=f'最优w: {w_list[min_loss_idx]:.2f}')
plt.xlabel('权重 w')
plt.ylabel('损失 Loss')
plt.title(f'当b={optimal_b:.2f}时，w与Loss的关系')
plt.grid(True)
plt.legend()

# 5. 可视化b与loss的关系（固定w为最优值附近）
optimal_w = w_list[min_loss_idx]

# 提取该w值下的b和对应的loss
fixed_w_b = []
fixed_w_loss = []
for w, b, loss_val in zip(w_list, b_list, mse_list):
    if abs(w - optimal_w) < 0.01:  # 找到最优w附近的值
        fixed_w_b.append(b)
        fixed_w_loss.append(loss_val)

# 绘制b与loss的关系图
plt.subplot(1, 2, 2)
plt.plot(fixed_w_b, fixed_w_loss, 'g-')
plt.axvline(x=optimal_b, color='r', linestyle='--', label=f'最优b: {optimal_b:.2f}')
plt.xlabel('偏置 b')
plt.ylabel('损失 Loss')
plt.title(f'当w={optimal_w:.2f}时，b与Loss的关系')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 6. 绘制3D曲面图，展示w、b与loss的关系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 将数据转换为网格形式
W, B = np.meshgrid(w_range, b_range)
Loss = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        w = W[i, j]
        b = B[i, j]
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, w, b)
        Loss[i, j] = l_sum / len(x_data)

# 绘制3D曲面
surf = ax.plot_surface(W, B, Loss, cmap='viridis', alpha=0.8)
# 标记最小损失点
ax.scatter([optimal_w], [optimal_b], [min(mse_list)], color='red', s=100, marker='*', label='最小损失点')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.set_title('w、b与Loss的关系曲面图')
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.legend()
plt.show()

# 输出最优的w和b值
print(f"\n最优参数:")
print(f"w = {optimal_w:.4f}")
print(f"b = {optimal_b:.4f}")
print(f"最小损失值 = {min(mse_list):.4f}")
