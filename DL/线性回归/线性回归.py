import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# 1. 读取数据
file_path = "train.csv"
try:
    df = pd.read_csv(file_path)
    print("数据读取成功！")
    print(f"数据集基本信息：{df.shape[0]}行，{df.shape[1]}列")
    print("\n原始数据前5行：")
    print(df.head())
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}，请检查文件路径是否正确")
    exit()
except Exception as e:
    print(f"读取数据时发生错误：{str(e)}")
    exit()

# 2. 缺失值分析与处理
print("\n=== 缺失值分析 ===")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失值数量': missing_values,
    '缺失比例(%)': missing_percentage.round(2)
})
print("各列缺失值情况：")
print(missing_df[missing_df['缺失值数量'] > 0])

# 选择合适的缺失值处理策略
# 对于数值型特征使用均值填充，类别型特征使用众数填充（根据实际数据类型调整）
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

# 处理数值型特征缺失值
if not numeric_cols.empty:
    num_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# 处理类别型特征缺失值（如果有的话）
if not categorical_cols.empty:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\n缺失值处理完成！")
print("处理后各列缺失值数量：")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 3. 准备训练数据（假设目标列为'y'，特征列为'x'，请根据实际数据列名修改）
# 自动检测可能的特征列和目标列（可根据实际情况手动指定）
target_col = 'y'  # 请替换为实际目标列名
feature_col = 'x'  # 请替换为实际特征列名

if target_col not in df.columns or feature_col not in df.columns:
    print(f"\n警告：未找到目标列'{target_col}'或特征列'{feature_col}'")
    print(f"可用列名：{df.columns.tolist()}")
    # 尝试使用最后一列作为目标列，第一列作为特征列
    target_col = df.columns[-1]
    feature_col = df.columns[0]
    print(f"自动选择目标列：{target_col}，特征列：{feature_col}")

X = df[[feature_col]].values
y = df[target_col].values

# 4. 训练线性回归模型 y = wx + b
model = LinearRegression()
model.fit(X, y)

# 获取模型参数
w = model.coef_[0]
b = model.intercept_
print(f"\n=== 模型参数 ===")
print(f"线性回归方程：y = {w:.4f}x + {b:.4f}")

# 5. 计算损失（均方误差）
y_pred = model.predict(X)
loss = np.mean((y - y_pred) **2)
print(f"模型在训练集上的均方误差（MSE）：{loss:.4f}")

# 6. 可视化结果
plt.figure(figsize=(15, 10))

# 子图1：数据散点图与拟合线
plt.subplot(2, 2, 1)
plt.scatter(X, y, alpha=0.5, label='原始数据')
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'拟合线: y = {w:.2f}x + {b:.2f}')
plt.xlabel(feature_col)
plt.ylabel(target_col)
plt.title('数据与线性拟合')
plt.legend()

# 子图2：w与loss的关系
plt.subplot(2, 2, 2)
w_values = np.linspace(w - 2*abs(w), w + 2*abs(w), 200) if w != 0 else np.linspace(-2, 2, 200)
loss_w = []
for w_sim in w_values:
    y_pred_sim = w_sim * X.flatten() + b
    loss_sim = np.mean((y - y_pred_sim)** 2)
    loss_w.append(loss_sim)
plt.plot(w_values, loss_w, 'g-')
plt.xlabel('w (斜率)')
plt.ylabel('损失 (MSE)')
plt.title('w与损失的关系')
plt.axvline(x=w, color='r', linestyle='--', label=f'最优w: {w:.4f}')
plt.legend()

# 子图3：b与loss的关系
plt.subplot(2, 2, 3)
b_values = np.linspace(b - 2*abs(b), b + 2*abs(b), 200) if b != 0 else np.linspace(-2, 2, 200)
loss_b = []
for b_sim in b_values:
    y_pred_sim = w * X.flatten() + b_sim
    loss_sim = np.mean((y - y_pred_sim) **2)
    loss_b.append(loss_sim)
plt.plot(b_values, loss_b, 'b-')
plt.xlabel('b (截距)')
plt.ylabel('损失 (MSE)')
plt.title('b与损失的关系')
plt.axvline(x=b, color='r', linestyle='--', label=f'最优b: {b:.4f}')
plt.legend()

# 子图4：残差图
plt.subplot(2, 2, 4)
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')

plt.tight_layout()
plt.show()
