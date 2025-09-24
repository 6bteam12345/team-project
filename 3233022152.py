import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
try:
    # 尝试从本地读取train.csv，如果不存在则使用示例数据
    data = pd.read_csv('train.csv')
    x_data = data['x'].values
    y_data = data['y'].values
    print("从train.csv读取数据成功")
    print(f"数据量: {len(x_data)}")
except:
    # 如果文件不存在，使用示例数据
    print("train.csv文件不存在，使用示例数据")
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([2.1, 3.9, 6.2, 8.1, 9.8])
    # 创建示例数据并保存为CSV
    example_data = pd.DataFrame({'x': x_data, 'y': y_data})
    example_data.to_csv('train.csv', index=False)
    print("已创建示例train.csv文件")


def forward(x, w, b):
    """前向传播计算预测值"""
    return x * w + b


def loss(x, y, w, b):
    """计算单个样本的损失"""
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


def total_loss(x_data, y_data, w, b):
    """计算所有样本的总损失（MSE）"""
    loss_sum = 0
    for x, y in zip(x_data, y_data):
        loss_sum += loss(x, y, w, b)
    return loss_sum / len(x_data)


# 测试不同的w和b值
w_range = np.arange(0.0, 4.1, 0.2)  # 减小步长以加快计算
b_range = np.arange(-2.0, 2.1, 0.2)

# 存储结果
results = []

print("开始训练...")
for i, w in enumerate(w_range):
    for j, b in enumerate(b_range):
        mse = total_loss(x_data, y_data, w, b)
        results.append((w, b, mse))
    print(f"进度: {((i + 1) / len(w_range)) * 100:.1f}%")

# 转换为numpy数组便于处理
results = np.array(results)
w_values = results[:, 0]
b_values = results[:, 1]
mse_values = results[:, 2]

best_idx = np.argmin(mse_values)
best_w = w_values[best_idx]
best_b = b_values[best_idx]
best_mse = mse_values[best_idx]

print(f"最佳参数: w={best_w:.2f}, b={best_b:.2f}, MSE={best_mse:.4f}")

# 创建图形
plt.figure(figsize=(15, 10))

# 子图1: w和loss之间的关系（固定b为最佳值）
plt.subplot(2, 2, 1)
# 筛选出b接近最佳值的点
mask_b = np.abs(b_values - best_b) < 0.1
w_filtered = w_values[mask_b]
mse_filtered = mse_values[mask_b]

# 按w排序
sort_idx = np.argsort(w_filtered)
w_sorted = w_filtered[sort_idx]
mse_sorted = mse_filtered[sort_idx]

plt.plot(w_sorted, mse_sorted, 'b-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Weight (w)')
plt.ylabel('Loss (MSE)')
plt.title(f'Weight vs Loss (b≈{best_b:.2f})')
plt.grid(True)

# 子图2: b和loss的关系（固定w为最佳值）
plt.subplot(2, 2, 2)
# 筛选出w接近最佳值的点
mask_w = np.abs(w_values - best_w) < 0.1
b_filtered = b_values[mask_w]
mse_filtered_b = mse_values[mask_w]

# 按b排序
sort_idx_b = np.argsort(b_filtered)
b_sorted = b_filtered[sort_idx_b]
mse_sorted_b = mse_filtered_b[sort_idx_b]

plt.plot(b_sorted, mse_sorted_b, 'r-', linewidth=2, marker='s', markersize=4)
plt.xlabel('Bias (b)')
plt.ylabel('Loss (MSE)')
plt.title(f'Bias vs Loss (w≈{best_w:.2f})')
plt.grid(True)

# 子图3: 散点图显示w, b和loss的关系
plt.subplot(2, 2, 3)
scatter = plt.scatter(w_values, b_values, c=mse_values, cmap='viridis',
                      s=20, alpha=0.7)
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.title('Weight vs Bias (颜色表示Loss)')
plt.colorbar(scatter, label='Loss (MSE)')
# 标记最佳点
plt.scatter(best_w, best_b, color='red', s=100, marker='*', label='最佳点')
plt.legend()

# 子图4: 拟合结果
plt.subplot(2, 2, 4)
plt.scatter(x_data, y_data, color='blue', label='真实数据', s=50, alpha=0.7)
x_line = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
y_line = forward(x_line, best_w, best_b)
plt.plot(x_line, y_line, color='red', linewidth=2,
         label=f'拟合直线: y={best_w:.2f}x+{best_b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('linear_regression_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 单独创建3D图（如果用户环境支持）
try:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 创建网格数据
    w_unique = np.unique(w_values)
    b_unique = np.unique(b_values)
    W, B = np.meshgrid(w_unique, b_unique)
    MSE = np.zeros_like(W)

    for i, w_val in enumerate(w_unique):
        for j, b_val in enumerate(b_unique):
            mask = (w_values == w_val) & (b_values == b_val)
            if np.any(mask):
                MSE[j, i] = mse_values[mask][0]

    surf = ax.plot_surface(W, B, MSE, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Bias (b)')
    ax.set_zlabel('Loss (MSE)')
    ax.set_title('Weight, Bias vs Loss (3D曲面)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 标记最佳点
    ax.scatter([best_w], [best_b], [best_mse], color='red', s=100, marker='*')

    plt.savefig('3d_loss_surface.png', dpi=300, bbox_inches='tight')
    plt.show()

except Exception as e:
    print(f"3D绘图失败: {e}")
    print("这可能是因为您的环境不支持3D绘图，但2D分析图已成功生成")

# 显示详细结果
print(f"\n=== 最终结果 ===")
print(f"最佳模型: y = {best_w:.4f} * x + {best_b:.4f}")
print(f"最小损失(MSE): {best_mse:.4f}")

print("\n=== 预测结果 ===")
print("x\t真实y\t预测y\t误差")
for i, (x, y_true) in enumerate(zip(x_data, y_data)):
    y_pred = forward(x, best_w, best_b)
    error = y_pred - y_true
    print(f"{x}\t{y_true:.2f}\t{y_pred:.2f}\t{error:+.2f}")

# 计算R²分数
y_mean = np.mean(y_data)
ss_tot = np.sum((y_data - y_mean) ** 2)
ss_res = np.sum((y_data - forward(x_data, best_w, best_b)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\n模型评估:")
print(f"R²分数: {r_squared:.4f}")
print(f"平均绝对误差: {np.mean(np.abs(y_data - forward(x_data, best_w, best_b))):.4f}")