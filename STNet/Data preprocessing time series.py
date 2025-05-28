import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from CBAM import CBAM  # Assuming the CBAM code is saved in cbam.py
from TCN import TemporalConvNet
import torch.nn.functional as F
import math
import csv
from sklearn.metrics import r2_score

'''
train_timedata的改进，改进了txt文件数据处理。
提取时间序列数据；
对这个300个数据点，使用60的滑动窗口提取数据，步长设置为10，提取得到这种浓度下的25组数据（数据大小则为60*12*25，解释为60秒、12个传感器、25组）

'''

def generate_labels(concentration_levels, num_points, data_length):
    labels = []
    for concentration in concentration_levels:
        labels.extend([concentration] * num_points)
    return labels[:data_length]

def process_data(file_path, concentrations_no, concentrations_no2, window_size, step_size):
    # Load data
    data = pd.read_csv(file_path, sep=',', header=None).values

    # Generate labels
    labels_no = generate_labels(concentrations_no, 300, data.shape[0])
    labels_no2 = generate_labels(concentrations_no2, 300, data.shape[0])
    labels = np.column_stack((labels_no, labels_no2))

    # Extract data using sliding window
    all_data, all_labels = [], []
    # 计算目标数据长度
    required_length = len(concentrations_no) * 300  #序列的实际长度

    for i in range(0, required_length - window_size + 1, step_size):
        window_data = data[i:i + window_size, :]
        middle_index = i + window_size // 2
        concentration_label = labels[middle_index]
        all_data.append(window_data)
        all_labels.append(concentration_label)
    '''
    这里标准化：
    ##  对每个txt文件数据全局特征标准化，，，起结果与特征全局标准化类似（）
    # 将 all_data 转换为 numpy 数组
    all_data = np.array(all_data)

    # 对 all_data 进行标准化
    # 重新调整形状以便对每个特征进行标准化处理
    reshaped_data = all_data.reshape(-1, all_data.shape[-1])  # 形状调整为 (num_samples * window_size, num_features)

    # 初始化 StandardScaler
    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(reshaped_data)  # 对所有数据进行标准化处理

    # 将标准化后的数据调整回原始的形状
    all_data_scaled = all_data_scaled.reshape(all_data.shape)  # 恢复为 (num_samples, window_size, num_features)

    '''

    return np.array(all_data), np.array(all_labels)

def combine_datasets(existing_data, existing_labels, new_data, new_labels):
    if existing_data.shape[1:] == new_data.shape[1:] and existing_labels.shape[1:] == new_labels.shape[1:]:
        combined_data = np.vstack((existing_data, new_data))
        combined_labels = np.vstack((existing_labels, new_labels))
        return combined_data, combined_labels
    else:
        raise ValueError("The shapes of the datasets do not match.")

# File paths and parameters
file_paths = [
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-1.txt',
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-2.txt',
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-1.txt',
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt',
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合20241122.txt',
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合前5.txt',
            'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合后.txt'
              ]
concentrations_no_list = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                          [0, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 70, 70, 80, 90, 0],
                          [0, 10, 20, 20, 30, 30, 0] ,
                          [0, 50, 60, 70, 80, 30, 40, 50, 80, 20, 0]
                          ]
concentrations_no2_list = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                           [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                           [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                           [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
                           [0, 20, 30, 10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 40, 10, 30, 20, 10, 0],
                           [0, 10, 10, 20, 10, 20, 0],
                           [0, 40, 30, 20, 10, 30, 40, 50, 20, 20, 0]
                           ]

window_size, step_size = 60, 10

# Initialize combined dataset
combined_X, combined_y = None, None

# Process each file
for file_path, concentrations_no, concentrations_no2 in zip(file_paths, concentrations_no_list, concentrations_no2_list):
    data, labels = process_data(file_path, concentrations_no, concentrations_no2, window_size, step_size)
    if combined_X is None:
        combined_X, combined_y = data, labels
    else:
        combined_X, combined_y = combine_datasets(combined_X, combined_y, data, labels)

# Output the combined dataset
print(f"Final combined data shape: {combined_X.shape}")
print(f"Final combined labels shape: {combined_y.shape}")

# Convert the combined data and labels to PyTorch tensors
X_tensor = torch.tensor(combined_X, dtype=torch.float32)  # Shape: (1986, 60, 12)
y_tensor = torch.tensor(combined_y, dtype=torch.float32)  # Shape: (1986, 2)

# 初始化一个存放标准化数据的张量
X_normalized = torch.zeros_like(X_tensor)

# 对每个样本单独进行标准化
for i in range(X_tensor.shape[0]):  # 遍历 2515 个样本
    scaler = MinMaxScaler()
    # 对当前样本 (60, 12) 的特征列进行标准化
    X_normalized[i] = torch.tensor(scaler.fit_transform(X_tensor[i].numpy()), dtype=torch.float32)

# Create a dataset
#dataset = TensorDataset(X_tensor, y_tensor)

# 1. 对数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_X.reshape(-1, combined_X.shape[-1]))  # 标准化每个特征列

# 2. 转换为 PyTorch tensor
#X_tensor = torch.tensor(X_scaled.reshape(combined_X.shape), dtype=torch.float32)
X_tensor = X_normalized
y_tensor = torch.tensor(combined_y, dtype=torch.float32)

# 3. 创建数据集
dataset = TensorDataset(X_tensor, y_tensor)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 60, 128)  # Flattened features size
        self.fc2 = nn.Linear(128, 2)       # Output size for two concentration labels (NO and NO2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length) for Conv1d
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, output_size)  # Second fully connected layer for output

    def forward(self, x):
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out has shape (batch_size, seq_len, hidden_size)
        # Take the last hidden state (can also be mean/average of all time steps)
        lstm_out = lstm_out[:, -1, :]  # We take the output of the last time step
        x = nn.ReLU()(self.fc1(lstm_out))
        x = self.fc2(x)
        return x

class CNNModelWithCBAM(nn.Module):
    def __init__(self):
        super(CNNModelWithCBAM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(in_channels=32, reduction=16, kernel_size=7)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(in_channels=64, reduction=16, kernel_size=7)

        self.fc1 = nn.Linear(64 * 60, 128)  # Adjust this based on the flattened features
        self.fc2 = nn.Linear(128, 2)       # Output size for two concentration labels (NO and NO2)

    def forward(self, x):
        # Permute input to match Conv1d expected shape
        x = x.permute(0, 2, 1)  # Shape: (batch_size, channels, sequence_length)

        # Apply first Conv1d + CBAM + ReLU
        x = nn.ReLU()(self.conv1(x))
        x = self.cbam1(x)

        # Apply second Conv1d + CBAM + ReLU
        x = nn.ReLU()(self.conv2(x))
        x = self.cbam2(x)

        # Flatten features for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, flattened_features)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModelWithTCN(nn.Module):
    def __init__(self):
        super(CNNModelWithTCN, self).__init__()
        # Define TCN layers
        self.tcn = TemporalConvNet(num_inputs=12, num_channels=[32, 64, 128], kernel_size=5, dropout=0.2)

        # Calculate the output size after TCN to correctly set the input size of the fully connected layer
        self._to_linear = self._calculate_conv_output_size()

        # Fully connected layers after the TCN layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 2)

    def _calculate_conv_output_size(self):
        # Use a dummy input to calculate the output size after TCN layers
        with torch.no_grad():
            dummy_input = torch.randn(1, 12, 60)  # Batch size 1, 14 channels, 48 time steps
            output = self.tcn(dummy_input)
            return output.view(-1).size(0)  # Flatten and get the number of features

    def forward(self, x):
        # 调整输入形状为 (batch_size, num_channels, seq_length)
        x = x.permute(0, 2, 1)  # (32, 60, 12) -> (32, 12, 60)

        x = self.tcn(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerEncoderTCNModel(nn.Module):
    def __init__(self, input_dim, seq_length, num_heads, num_layers, hidden_dim, tcn_channels, output_dim,
                 kernel_size=2, dropout=0.2):
        super(TransformerEncoderTCNModel, self).__init__()

        # Transformer Encoder 部分
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self._generate_positional_encoding(seq_length, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # TCN 部分
        self.tcn = TemporalConvNet(num_inputs=hidden_dim, num_channels=tcn_channels, kernel_size=kernel_size,
                                   dropout=dropout)

        # 输出层
        tcn_out_dim = tcn_channels[-1] * seq_length
        self.fc1 = nn.Linear(tcn_out_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def _generate_positional_encoding(self, seq_length, model_dim):
        """
        生成位置编码矩阵
        :param seq_length: 序列长度
        :param model_dim: 模型隐藏层维度
        :return: 位置编码 (1, seq_length, model_dim)
        """
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)  # (seq_length, 1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(seq_length, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_length, model_dim)

    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_dim)
        batch_size, seq_length, input_dim = x.size()

        # Transformer Encoder 部分
        x = self.embedding(x)  # (batch_size, seq_length, hidden_dim)
        x = x + self.positional_encoding.to(x.device)  # 加入位置编码
        x = x.permute(1, 0, 2)  # 转换形状以适配 Transformer: (seq_length, batch_size, hidden_dim)
        x = self.transformer_encoder(x)  # (seq_length, batch_size, hidden_dim)
        x = x.permute(1, 2, 0)  # 转换形状以适配 TCN: (batch_size, hidden_dim, seq_length)

        # TCN 部分
        x = self.tcn(x)  # (batch_size, tcn_channels[-1], seq_length)
        x = x.view(batch_size, -1)  # 展平: (batch_size, tcn_channels[-1] * seq_length)

        # 输出部分
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
#model = CNNModelWithCBAM().to(device)
#model = LSTMModel(input_size=12, hidden_size=64, num_layers=2, output_size=2).to(device)
#model = CNNModelWithTCN().to(device)

# 定义模型参数
input_dim = 12         # 输入特征维度 (传感器数量)
seq_length = 60        # 序列长度
num_heads = 8          # Transformer 注意力头数量
num_layers = 4         # Transformer 层数
hidden_dim = 64        # Transformer 隐藏层维度
tcn_channels = [32, 64]  # TCN 通道列表
kernel_size = 5        # TCN 卷积核大小
output_dim = 2         # 输出维度 (例如: NO 和 NO2 的浓度预测)
dropout = 0.2          # Dropout 概率

# 初始化模型
#model = TransformerEncoderTCNModel(input_dim, seq_length, num_heads, num_layers, hidden_dim, tcn_channels, output_dim, kernel_size, dropout)

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Initialize lists to store loss values
train_losses = []
val_losses = []

# Training loop
num_epochs = 300
test_r2_scores = []  # 用于保存测试集的 R^2 系数

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)  # Record train loss

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)  # Record validation loss

    # Test set evaluation
    test_predictions = []
    test_true_values = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:  # Assuming you have a test_loader
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_predictions.append(outputs.cpu().numpy())
            test_true_values.append(y_batch.cpu().numpy())

    # Concatenate all predictions and true values
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_true_values = np.concatenate(test_true_values, axis=0)

    # Calculate R² score for the test set
    test_r2 = r2_score(test_true_values, test_predictions)
    test_r2_scores.append(test_r2)  # Record R² score for this epoch

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save loss values and R² scores to a CSV file
loss_file = "losses.csv"
with open(loss_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Test R2"])
    for epoch, (train_loss, val_loss, test_r2) in enumerate(zip(train_losses, val_losses, test_r2_scores), 1):
        writer.writerow([epoch, train_loss, val_loss, test_r2])

print(f"Loss values and Test R² scores saved to {loss_file}")


# Evaluate model on validation set
model.eval()
predictions = []
true_values = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        predictions.append(outputs.cpu().numpy())
        true_values.append(y_batch.cpu().numpy())

# Concatenate all predictions and true values
import numpy as np
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Calculate metrics (e.g., R² score)
from sklearn.metrics import r2_score
r2 = r2_score(true_values, predictions)
print(f"R² Score: {r2:.4f}")

import csv
import numpy as np

# 假设 true_values 和 predictions 是你的真实值和预测值 numpy 数组
# true_values 和 predictions 的形状为 (样本数, 2)，分别表示 NO 和 NO2 的真实值与预测值

# 创建一个字典来保存每个真实浓度对应的预测值
true_concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 数据按真实浓度分组
data_dict = {concentration: [] for concentration in true_concentrations}

for true, pred in zip(true_values[:, 0], predictions[:, 0]):  # 针对 NO 的数据
    if true in data_dict:
        data_dict[true].append(pred)

# 将数据格式化为您需要的形式
output_data = []
max_len = max(len(data_dict[key]) for key in data_dict.keys())  # 找出最多的预测值数
for i in range(max_len):
    row = []
    for concentration in true_concentrations:
        if i < len(data_dict[concentration]):
            row.append(data_dict[concentration][i])  # 预测值
        else:
            row.append("")  # 如果没有更多预测值，用空白填充
    output_data.append(row)

# 保存为 CSV 文件
output_file = "true_vs_predicted_formatted.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    # 写入标题
    header = []
    for concentration in true_concentrations:
        header.append(f"True_{concentration}")
        header.append(f"Predicted_{concentration}")
    writer.writerow(header)

    # 写入数据
    for i in range(len(output_data)):
        row = []
        for j, concentration in enumerate(true_concentrations):
            row.append(concentration)  # 添加真实浓度
            row.append(output_data[i][j])  # 添加预测值
        writer.writerow(row)

print(f"数据已保存到 {output_file}")




# Plot loss curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.scatter(true_values[:, 0], predictions[:, 0], label="NO", alpha=0.6)
plt.scatter(true_values[:, 1], predictions[:, 1], label="NO2", alpha=0.6)
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("True Concentration")
plt.ylabel("Predicted Concentration")
plt.legend()
plt.title("True vs Predicted Concentrations")
plt.show()

# Plot scatter plots for NO and NO2
plt.figure(figsize=(14, 6))

# NO Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(true_values[:, 0], predictions[:, 0], alpha=0.6, label="Predictions")
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("True NO Concentration")
plt.ylabel("Predicted NO Concentration")
plt.title("True vs Predicted NO")
plt.legend()
plt.grid(True)

# NO2 Scatter Plot
plt.subplot(1, 2, 2)
plt.scatter(true_values[:, 1], predictions[:, 1], alpha=0.6, label="Predictions")
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("True NO2 Concentration")
plt.ylabel("Predicted NO2 Concentration")
plt.title("True vs Predicted NO2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.metrics import r2_score, mean_absolute_error

# Evaluate model on validation set
model.eval()
predictions = []
true_values = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        predictions.append(outputs.cpu().numpy())
        true_values.append(y_batch.cpu().numpy())

# Concatenate all predictions and true values
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Calculate R² and MAE for NO and NO2
r2_no = r2_score(true_values[:, 0], predictions[:, 0])
r2_no2 = r2_score(true_values[:, 1], predictions[:, 1])
mae_no = mean_absolute_error(true_values[:, 0], predictions[:, 0])
mae_no2 = mean_absolute_error(true_values[:, 1], predictions[:, 1])

print(f"R² (NO): {r2_no:.4f}, R² (NO2): {r2_no2:.4f}")
print(f"MAE (NO): {mae_no:.4f}, MAE (NO2): {mae_no2:.4f}")

# Plot true vs predicted concentrations for NO and NO2
for i, gas_name in enumerate(['NO', 'NO2']):
    # 对 y_test 的当前气体浓度列排序
    sorted_indices = np.argsort(true_values[:, i])
    y_test_sorted = true_values[sorted_indices, i]
    y_pred_sorted = predictions[sorted_indices, i]

    plt.figure(figsize=(10, 6))
    plt.scatter(true_values[:, i], predictions[:, i], alpha=0.5)
    plt.plot([min(true_values[:, i]), max(true_values[:, i])], [min(true_values[:, i]), max(true_values[:, i])], 'r--')
    plt.xlabel(f'True {gas_name} Concentration')
    plt.ylabel(f'Predicted {gas_name} Concentration')
    plt.title(f'True vs Predicted {gas_name} Concentration')
    plt.grid(True)
    plt.show()

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_sorted, label='True Concentration', color='blue', marker='o')
    plt.plot(y_pred_sorted, label='Predicted Concentration', color='red', marker='x')
    plt.xlabel('Test Sample (Sorted)')
    plt.ylabel('Concentration')
    plt.title(f'True vs Predicted {gas_name} Concentration (Sorted by True Values)')
    plt.legend()
    plt.grid(True)
    plt.show()