
# Updated Python code with dual outputs for NO and NO2 concentrations

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# File paths of the .txt files
file_paths = [
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-1.txt',
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-2.txt',
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-1.txt',
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'
    # Add more files as needed
]

# Initialize empty list to store all data
all_data = []
all_labels = []

# Label generation for the concentration levels
concentrations_no = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0]  # Concentration levels for NO
concentrations_no2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0]  # Concentration levels for NO2

# Loop through each file
for file_path in file_paths:
    # Load the data
    data = pd.read_csv(file_path, sep=',', header=None)

    # Initialize labels based on file type
    if 'no-' in file_path.lower():  # If the file is NO data
        labels_no = []
        labels_no2 = [0] * 3600  # NO2 concentration is always 0 in NO files
        for c_no in concentrations_no:
            labels_no.extend([c_no] * 300)  # Repeat each NO concentration value 300 times
        labels_no = labels_no[:data.shape[0]]  # Slice to match data length
    elif 'no2-' in file_path.lower():  # If the file is NO2 data
        labels_no = [0] * 3600  # NO concentration is always 0 in NO2 files
        labels_no2 = []
        for c_no2 in concentrations_no2:
            labels_no2.extend([c_no2] * 300)  # Repeat each NO2 concentration value 300 times
        labels_no2 = labels_no2[:data.shape[0]]  # Slice to match data length
    else:
        raise ValueError(f"Unrecognized file format: {file_path}")

    # Combine NO and NO2 labels into a single array
    labels = np.column_stack((labels_no, labels_no2))

    # Normalize the sensor data (scaling to range [0, 1])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Append data and labels to the respective lists
    all_data.append(data[1:3601])  # Ensure consistent data length
    all_labels.append(labels[:3600])  # Ensure consistent label length

# Convert the list of data and labels into a single array
X1 = np.vstack(all_data)  # Combine all data into one array
y1 = np.vstack(all_labels)  # Combine all labels into one array

# File paths of the .txt files
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合20241122.txt'
    # Add more files as needed


# Initialize empty list to store all data
all_data = []
all_labels = []

# Label generation for the concentration levels
concentrations_no = [0, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 70, 70, 80, 90, 0]  # Concentration levels for NO
concentrations_no2 = [0, 20, 30, 10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 40, 10, 30, 20, 10, 0]  # Concentration levels for NO2

# Load the data
data = pd.read_csv(file_path, sep=',', header=None)

# Initialize labels based on file type

labels_no = []
labels_no2 = []  # NO2 concentration is always 0 in NO files

for c_no in concentrations_no:
    labels_no.extend([c_no] * 300)  # Repeat each NO concentration value 300 times
labels_no = labels_no[:data.shape[0]]  # Slice to match data length
for c_no2 in concentrations_no2:
    labels_no2.extend([c_no2] * 300)  # Repeat each NO concentration value 300 times
labels_no2 = labels_no2[:data.shape[0]]  # Slice to match data length

# Combine NO and NO2 labels into a single array
labels = np.column_stack((labels_no, labels_no2))

# Normalize the sensor data (scaling to range [0, 1])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Append data and labels to the respective lists
all_data.append(data[1:5701])  # Ensure consistent data length
all_labels.append(labels[:])  # Ensure consistent label length

# Convert the list of data and labels into a single array
X2 = np.vstack(all_data)  # Combine all data into one array
y2 = np.vstack(all_labels)  # Combine all labels into one array

# 将数据垂直堆叠，合并为一个新的数据集
X = np.vstack((X1, X2))
y = np.vstack((y1, y2))

# File paths of the .txt files
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合前5.txt'
    # Add more files as needed


# Initialize empty list to store all data
all_data = []
all_labels = []

# Label generation for the concentration levels
concentrations_no = [0, 10, 20, 20, 30, 30, 0]  # Concentration levels for NO
concentrations_no2 = [0, 10, 10, 20, 10, 20, 0]  # Concentration levels for NO2

# Load the data
data = pd.read_csv(file_path, sep=',', header=None)

# Initialize labels based on file type

labels_no = []
labels_no2 = []  # NO2 concentration is always 0 in NO files

for c_no in concentrations_no:
    labels_no.extend([c_no] * 300)  # Repeat each NO concentration value 300 times
labels_no = labels_no[:data.shape[0]]  # Slice to match data length
for c_no2 in concentrations_no2:
    labels_no2.extend([c_no2] * 300)  # Repeat each NO concentration value 300 times
labels_no2 = labels_no2[:data.shape[0]]  # Slice to match data length

# Combine NO and NO2 labels into a single array
labels = np.column_stack((labels_no, labels_no2))

# Normalize the sensor data (scaling to range [0, 1])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Append data and labels to the respective lists
all_data.append(data[1:2101])  # Ensure consistent data length
all_labels.append(labels[:])  # Ensure consistent label length

# Convert the list of data and labels into a single array
X3 = np.vstack(all_data)  # Combine all data into one array
y3 = np.vstack(all_labels)  # Combine all labels into one array

# File paths of the .txt files
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合后.txt'
    # Add more files as needed


# Initialize empty list to store all data
all_data = []
all_labels = []

# Label generation for the concentration levels
concentrations_no = [0, 50, 60, 70, 80, 30, 40, 50, 80, 20, 0]  # Concentration levels for NO
concentrations_no2 = [0, 40, 30, 20, 10, 30, 40, 50, 20, 20, 0]  # Concentration levels for NO2

# Load the data
data = pd.read_csv(file_path, sep=',', header=None)

# Initialize labels based on file type

labels_no = []
labels_no2 = []  # NO2 concentration is always 0 in NO files

for c_no in concentrations_no:
    labels_no.extend([c_no] * 300)  # Repeat each NO concentration value 300 times
labels_no = labels_no[:data.shape[0]]  # Slice to match data length
for c_no2 in concentrations_no2:
    labels_no2.extend([c_no2] * 300)  # Repeat each NO concentration value 300 times
labels_no2 = labels_no2[:data.shape[0]]  # Slice to match data length

# Combine NO and NO2 labels into a single array
labels = np.column_stack((labels_no, labels_no2))

# Normalize the sensor data (scaling to range [0, 1])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Append data and labels to the respective lists
all_data.append(data[1:3301])  # Ensure consistent data length
all_labels.append(labels[:])  # Ensure consistent label length

# Convert the list of data and labels into a single array
X4 = np.vstack(all_data)  # Combine all data into one array
y4 = np.vstack(all_labels)  # Combine all labels into one array

# 将数据垂直堆叠，合并为一个新的数据集
X = np.vstack((X1, X2, X3 , X4))
y = np.vstack((y1, y2 ,y3 , y4))

# Normalize the sensor data (scaling to range [0, 1])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Check the shape of X and y to ensure they are correct
print(f"X shape: {X.shape}")  # Should be (total_samples, 12)
print(f"y shape: {y.shape}")  # Should be (total_samples, 2)
print(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the deep neural network model with dual outputs
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.fc4(x)
        return x


# Hyperparameters
input_size = 12  # Number of sensor features (columns)
hidden_sizes = [128, 256, 64, 32]  # Size of hidden layers
output_size = 2  # Dual outputs: NO and NO2 concentrations
learning_rate = 0.001
num_epochs = 3000

# Initialize the model, loss function, and optimizer
model = DNNModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
criterion = nn.L1Loss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training and test losses for plotting
train_losses = []
test_losses = []

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train_tensor)

    # Compute the loss
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Save the training loss for plotting
    train_losses.append(loss.item())

    # Evaluate the model on the test set after each epoch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        test_loss = criterion(y_pred_test, y_test_tensor)

    # Save the test loss for plotting
    test_losses.append(test_loss.item())

    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Plot the training and test loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.grid(True)
plt.legend()
plt.show()

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = criterion(y_pred_test, y_test_tensor)
    print(f"Test Loss (final): {test_loss.item():.4f}")

# Convert predictions and true values to numpy for comparison
y_pred_test = y_pred_test.numpy()
y_test = y_test_tensor.numpy()

# Compute the R² score for each output
r2_no = r2_score(y_test[:, 0], y_pred_test[:, 0])
r2_no2 = r2_score(y_test[:, 1], y_pred_test[:, 1])
print(f"R² Score for NO: {r2_no:.4f}")
print(f"R² Score for NO2: {r2_no2:.4f}")

mae_no = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])  # NO
mae_no2 = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])  # NO2
print(f"MAE for NO: {mae_no}, MAE for NO2: {mae_no2}")

# Plot true vs predicted concentrations for NO and NO2
for i, gas_name in enumerate(['NO', 'NO2']):
    # 对 y_test 的当前气体浓度列排序
    sorted_indices = np.argsort(y_test[:, i])
    y_test_sorted = y_test[sorted_indices, i]
    y_pred_sorted = y_pred_test[sorted_indices, i]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, i], y_pred_test[:, i], alpha=0.5)
    plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], 'r--')
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
