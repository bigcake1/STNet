import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader

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
X = np.vstack(all_data)  # Combine all data into one array
y = np.vstack(all_labels)  # Combine all labels into one array

# Check the shape of X and y to ensure they are correct
print(f"X shape: {X.shape}")  # Should be (total_samples, 12)
print(f"y shape: {y.shape}")  # Should be (total_samples, 2)
print(y)