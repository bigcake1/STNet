import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from CBAM import CBAM  # Assuming the CBAM code is saved in cbam.py
from  CBAM import ChannelAttention
from TCN import TemporalConvNet
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
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
    scaler = StandardScaler()
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


class CNNModelWithCBAM(nn.Module):
    def __init__(self):
        super(CNNModelWithCBAM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(in_channels=32, reduction=16, kernel_size=7)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(in_channels=64, reduction=16, kernel_size=7)

        # Define TCN layers
        self.tcn = TemporalConvNet(num_inputs=64, num_channels=[12, 64, 128], kernel_size=5, dropout=0.2)

        # Calculate the output size after TCN to correctly set the input size of the fully connected layer
        self._to_linear = self._calculate_conv_output_size()

        # Fully connected layers after the TCN layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 2)

    def _calculate_conv_output_size(self):
        # Use a dummy input to calculate the output size after TCN layers
        with torch.no_grad():
            dummy_input = torch.randn(1, 64, 60)  # Batch size 1, 14 channels, 48 time steps
            output = self.tcn(dummy_input)
            return output.view(-1).size(0)  # Flatten and get the number of features

    def forward(self, x):
        # Permute input to match Conv1d expected shape
        x = x.permute(0, 2, 1)  # Shape: (batch_size, channels, sequence_length)

        # Apply first Conv1d + CBAM + ReLU
        x = nn.ReLU()(self.conv1(x))
        x = self.cbam1(x)

        # Apply second Conv1d + CBAM + ReLU
        x = nn.ReLU()(self.conv2(x))
        x = self.cbam2(x)

        # 调整输入形状为 (batch_size, num_channels, seq_length)
        x = x.permute(0, 1, 2)  #
        x = self.tcn(x)

        # Flatten features for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, flattened_features)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the model parameters
input_dim = 12  # For example, the number of sensors (adjust based on your data)
seq_length = 60  # Length of each input sequence (adjust as needed)
num_heads = 8  # Number of attention heads (can be adjusted based on your needs)
num_layers = 4  # Number of layers (optional, not directly used in this model)
hidden_dim = 64  # Hidden dimension for the initial Conv1d layer
tcn_channels = [64, 128, 256]  # Number of channels for TCN layers
output_dim = 1  # Output dimension, e.g., for regression output (can be adjusted for classification)
kernel_size = 2  # Kernel size for TCN layers
dropout = 0.2  # Dropout rate for TCN layers
reduction = 16  # Reduction factor for the CBAM module
cbam_kernel_size = 7  # Kernel size for CBAM Spatial Attention

# Instantiate the model
model = CNNModelWithCBAM()


#model = CNNModelWithTCN().to(device)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store loss values
train_losses = []
val_losses = []

# Training loop
num_epochs = 200
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

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


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

import matplotlib.pyplot as plt

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