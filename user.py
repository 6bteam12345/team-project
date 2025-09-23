
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ========== 字体设置 ==========
font_list = [
    'Microsoft YaHei',
    'SimHei',
    'Arial Unicode MS',
    'WenQuanYi Zen Hei',
    'DejaVu Sans'  # 英文备用字体
]

# 检查系统可用字体
available_fonts = set(f.name for f in fm.fontManager.ttflist)
print("系统可用字体:", available_fonts)

# 设置实际使用字体
used_font = None
for font in font_list:
    if font in available_fonts:
        used_font = font
        break

if used_font:
    plt.rcParams['font.sans-serif'] = [used_font]
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("警告：未找到中文字体，将使用英文显示")


# ========== 线性回归模型 ==========
class LinearRegression:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x):
        return x * self.w + self.b

    def train(self, x_data, y_data, lr=0.01, epochs=100):
        history = []
        for epoch in range(epochs):
            y_pred = self.forward(x_data)
            dw = np.mean(2 * (y_pred - y_data) * x_data)
            db = np.mean(2 * (y_pred - y_data))
            self.w -= lr * dw
            self.b -= lr * db
            loss = np.mean((y_pred - y_data) ** 2)
            history.append({'w': self.w, 'b': self.b, 'loss': loss})
        return pd.DataFrame(history)


# ========== 数据加载 ==========
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("\n=== 数据列名 ===")
        print(df.columns.tolist())
        print("\n=== 前5行数据 ===")
        print(df.head())
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None


# ========== 可视化 ==========
def visualize(history):
    plt.figure(figsize=(14, 6))

    # 第一幅图：w-loss关系
    plt.subplot(1, 2, 1)
    plt.plot(history['w'], history['loss'], 'b-', linewidth=2)
    plt.title('权重(w)与损失(loss)的关系', fontsize=12)
    plt.xlabel('权重(w)', fontsize=10)
    plt.ylabel('损失(loss)', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 第二幅图：b-loss关系
    plt.subplot(1, 2, 2)
    plt.plot(history['b'], history['loss'], 'r-', linewidth=2)
    plt.title('偏置(b)与损失(loss)的关系', fontsize=12)
    plt.xlabel('偏置(b)', fontsize=10)
    plt.ylabel('损失(loss)', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片（尝试不同格式）
    for ext in ['.png', '.pdf', '.svg']:
        try:
            plt.savefig(f'loss_curves{ext}', dpi=300, bbox_inches='tight')
            print(f"图表已保存为 loss_curves{ext}")
            break
        except Exception as e:
            print(f"保存{ext}格式失败: {e}")

    plt.show()


# ========== 主程序 ==========
if __name__ == "__main__":
    # 配置参数
    data_path = r'E:\train.csv'  # 替换为您的实际路径
    student_id = '123456'  # 替换为您的学号

    # 加载数据
    x, y = load_data(data_path)

    if x is not None:
        # 训练模型
        print("\n开始训练...")
        model = LinearRegression()
        history_df = model.train(x, y, epochs=200)

        # 可视化
        visualize(history_df)

        # 保存结果
        history_df.to_csv('training_history.csv', index=False)
        print("\n训练结果已保存到 training_history.csv")
        print(f"学号: {3233054221}")
    else:
        print("程序终止：数据加载失败")


