import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CountriesDataset(Dataset):
    def __init__(self, filepath, target_column='Total Ecological Footprint'):
        data = pd.read_csv(filepath)
        data = self.preprocess_data(data)

        feature_columns = [col for col in data.columns if col != target_column]
        x_data = data[feature_columns].values.astype(np.float32)
        y_data = data[target_column].values.astype(np.float32)

        # 标准化
        self.x_mean = x_data.mean(axis=0)
        self.x_std = x_data.std(axis=0)
        self.y_mean = y_data.mean()
        self.y_std = y_data.std()

        x_data = (x_data - self.x_mean) / (self.x_std + 1e-8)
        y_data = (y_data - self.y_mean) / (self.y_std + 1e-8)

        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
        self.len = len(data)

    def preprocess_data(self, data):
        processed_data = data.copy()
        for column in processed_data.columns:
            if processed_data[column].dtype == 'object':
                if processed_data[column].astype(str).str.contains('\$', na=False).any():
                    temp_col = processed_data[column].astype(str).str.replace('$', '', regex=False)
                    temp_col = temp_col.str.replace(',', '', regex=False)
                    temp_col = temp_col.str.replace('"', '', regex=False)
                    processed_data[column] = pd.to_numeric(temp_col, errors='coerce')
                else:
                    unique_vals = processed_data[column].dropna().unique()
                    val_to_num = {val: i for i, val in enumerate(unique_vals)}
                    processed_data[column] = processed_data[column].map(val_to_num)

            if processed_data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(processed_data[column]):
                    processed_data[column] = processed_data[column].fillna(processed_data[column].median())
                else:
                    processed_data[column] = processed_data[column].fillna(0)
        return processed_data

    def inverse_transform_y(self, y_normalized):
        return y_normalized * self.y_std + self.y_mean

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class FiveLayerNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(FiveLayerNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 7)
        self.layer2 = torch.nn.Linear(7, 6)
        self.layer3 = torch.nn.Linear(6, 5)
        self.layer4 = torch.nn.Linear(5, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x


def train_and_save_model():
    # 创建数据集
    dataset = CountriesDataset('countries.csv')

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 创建模型
    input_size = dataset.x_data.shape[1]
    model = FiveLayerNetwork(input_size)

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # 训练循环
    for epoch in range(100):
        # 训练
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 保存最佳模型（只保存模型状态）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/100], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print(f'\n训练完成! 最佳验证损失: {best_val_loss:.4f}')

    # 重新创建模型并加载权重
    model = FiveLayerNetwork(input_size)
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))

    # 测试
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # 处理输出形状
            if outputs.dim() > 1:
                predictions.extend(outputs.squeeze().tolist())
            else:
                predictions.extend(outputs.tolist())
            actuals.extend(labels.tolist())

    # 转换回原始尺度
    predictions_original = dataset.inverse_transform_y(np.array(predictions))
    actuals_original = dataset.inverse_transform_y(np.array(actuals))

    # 可视化
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)

    # 预测vs实际值
    plt.subplot(1, 2, 2)
    plt.scatter(actuals_original, predictions_original, alpha=0.6)
    min_val = min(actuals_original.min(), predictions_original.min())
    max_val = max(actuals_original.max(), predictions_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 实际值')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 计算评估指标
    mse = np.mean((predictions_original - actuals_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_original - actuals_original))

    print(f'\n测试集结果:')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'最佳模型已保存为: best_model.pt')


if __name__ == '__main__':
    train_and_save_model()