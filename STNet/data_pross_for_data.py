'''
提取时间序列数据；
对这个300个数据点，使用60的滑动窗口提取数据，步长设置为10，提取得到这种浓度下的25组数据（数据大小则为60*12*25，解释为60秒、12个传感器、25组）
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define file paths of the .txt files
file_paths = [
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-1.txt',
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-2.txt',
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-1.txt',
    'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'
    # Add more files as needed
]

# Concentration levels for NO and NO2 (each concentration lasts 300 data points)
concentrations_no = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0]  # Concentration levels for NO
concentrations_no2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0]  # Concentration levels for NO2

# Initialize empty list to store all data and labels
all_data = []
all_labels = []

# Sliding window parameters
window_size = 60  # Window size of 60 seconds (data points)
step_size = 10    # Step size of 10 (each slide will shift by 10 points)
num_samples = 25  # Number of samples to be extracted

# Loop through each file to process the data
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

    data_scaled = np.array(data)

    # Extract data using sliding window
    for i in range(0, 3600 - window_size + 1, step_size):
        # For each concentration, extract the 60 second window and add to dataset
        window_data = data_scaled[i:i + window_size, :]  # Extract a 60-seconds window of sensor data
        # The label is the concentration of the middle data point in the window
        middle_index = i + window_size // 2  # Middle index of the window
        concentration_label = labels[middle_index]  # Get the concentration label at the middle of the window

        # Append the extracted window and the corresponding label
        all_data.append(window_data)
        all_labels.append(concentration_label)

# Convert the list of data and labels into a single array
X1 = np.array(all_data)  # Shape (num_samples, 60, 12)
y1 = np.array(all_labels)  # Shape (num_samples, 2) (NO and NO2 concentrations)

# Ensure the dataset has the required number of samples (should be 25 samples per concentration)
print(f"Shape of X1: {X1.shape}")
print(f"Shape of y1: {y1.shape}")

# File path
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合20241122.txt'

# Parameters
window_size = 60  # Sliding window size
step_size = 10    # Step size for sliding window

# Label generation for the concentration levels
concentrations_no = [0, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 70, 70, 80, 90, 0]  # NO concentrations
concentrations_no2 = [0, 20, 30, 10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 40, 10, 30, 20, 10, 0]  # NO2 concentrations

# Load the data
data = pd.read_csv(file_path, sep=',', header=None).values

# Generate labels for NO and NO2
labels_no = []
labels_no2 = []
for c_no in concentrations_no:
    labels_no.extend([c_no] * 300)  # Repeat each NO concentration value 300 times
labels_no = labels_no[:data.shape[0]]  # Slice to match data length

for c_no2 in concentrations_no2:
    labels_no2.extend([c_no2] * 300)  # Repeat each NO2 concentration value 300 times
labels_no2 = labels_no2[:data.shape[0]]  # Slice to match data length

# Combine NO and NO2 labels into a single array
labels = np.column_stack((labels_no, labels_no2))

# Normalize the sensor data (scaling to range [0, 1])
data_scaled = np.array(data)

# Initialize lists to store sliding window samples and corresponding labels
all_data = []
all_labels = []

# Extract data using sliding window
for i in range(0, data.shape[0] - window_size + 1, step_size):
    # Extract a 60-seconds window of sensor data
    window_data = data_scaled[i:i + window_size, :]  # Shape: (60, 12)

    # The label is the concentration of the middle data point in the window
    middle_index = i + window_size // 2  # Middle index of the window
    concentration_label = labels[middle_index]  # Get the concentration label at the middle of the window

    # Append the extracted window and the corresponding label
    all_data.append(window_data)
    all_labels.append(concentration_label)

# Convert lists to arrays for easier processing
all_data = np.array(all_data)  # Shape: (num_samples, 60, 12)
all_labels = np.array(all_labels)  # Shape: (num_samples, 2)

# Output the results
print(f"Processed data shape: {all_data.shape}")  # (num_samples, 60, 12)
print(f"Processed labels shape: {all_labels.shape}")  # (num_samples, 2)


# 假设 X1, y1 是第一个数据集的特征和标签
# 假设 processed_data, processed_labels 是第二个数据集的特征和标签

# 检查数据维度是否一致
if X1.shape[1:] == all_data.shape[1:] and y1.shape[1:] == all_labels.shape[1:]:
    # 合并特征和标签
    combined_X = np.vstack((X1, all_data))  # 合并特征
    combined_y = np.vstack((y1, all_labels))  # 合并标签

    # 输出合并后数据的形状
    print(f"Combined data shape: {combined_X.shape}")  # (num_samples, 60, 12)
    print(f"Combined labels shape: {combined_y.shape}")  # (num_samples, 2)
else:
    print("The shapes of the datasets do not match and cannot be combined.")

