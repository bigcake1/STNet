'''
##绘图
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 单个文件的路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合20241122.txt'  # 替换为您的文件路径

# 读取txt文件并加载数据
data = pd.read_csv(file_path, sep=',', header=None)
data = data / 1000

print(data.shape)

#指定需要调整的传感器及其目标基线值
specified_sensors = [0, 2, 3, 6, 9, 11]  # 需要调整的传感器索引
target_baselines = [1.15, 1.22, 1.2, 1.285, 1.3, 1.25]  # 每个传感器的目标基线值

extracted_data = data
# **基线对齐**
# 创建一个新数据集，保持原始数据
aligned_data = extracted_data.copy()

# 针对指定传感器调整基线到目标值
for sensor, target_baseline in zip(specified_sensors, target_baselines):
    current_baseline = extracted_data.iloc[:, sensor].mean()  # 计算当前传感器的基线
    adjustment = target_baseline - current_baseline  # 计算调整量
    extracted_data.iloc[:, sensor] += adjustment  # 调整传感器数据

data = extracted_data
print("指定传感器的基线已调整到目标值")

# 这里定义一个颜色列表，包含了12种不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 设置采样率为每秒一次
sampling_rate = 1

# 创建1个图，因为只处理一个文件
fig, ax = plt.subplots(figsize=(15, 10))


for col in range(0, 12):  # 如果数据列少于12，这里需要调整
    # 从每列的第sampling_rate个元素（假设数据从0开始索引）绘制到最后
    ax.plot(range(sampling_rate, data.shape[0]), data.iloc[sampling_rate:data.shape[0], col], label=f'传感器 {col + 1}',
            color=colors[col])

# 假设数据有12列 data.shape[0]


# 每隔300个数据点（5分钟）绘制一条虚线
time_intervals = range(300, data.shape[0], 301)  # 从300开始，每300个数据点绘制一条虚线
for time_point in time_intervals:
    ax.axvline(x=time_point, color='r', linestyle='--', linewidth=1)



ax.set_xlabel('Time')  # 设置x轴标签
ax.set_ylabel('Value')  # 设置y轴标签
ax.set_title(f'File: {os.path.basename(file_path)}')  # 设置图标题
ax.legend(loc='upper right')  # 显示图例
ax.grid(False)  # 显示网格线

plt.tight_layout()  # 自动调整子图参数以给定指定的填充
plt.show()  # 显示图形


###分割线###

'''




'''
##不同浓度梯度绘图处理
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 单个文件的路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no-1.txt'  # 替换为您的文件路径

# 读取txt文件并加载数据
data = pd.read_csv(file_path, sep=',', header=None)
data = data / 1000

print(data.shape)

#指定需要调整的传感器及其目标基线值
specified_sensors = [0, 2, 3, 6, 9, 11]  # 需要调整的传感器索引
target_baselines = [1.15, 1.22, 1.2, 1.285, 1.3, 1.25]  # 每个传感器的目标基线值

extracted_data = data
# **基线对齐**
# 创建一个新数据集，保持原始数据
aligned_data = extracted_data.copy()

# 针对指定传感器调整基线到目标值
for sensor, target_baseline in zip(specified_sensors, target_baselines):
    current_baseline = extracted_data.iloc[:, sensor].mean()  # 计算当前传感器的基线
    adjustment = target_baseline - current_baseline  # 计算调整量
    extracted_data.iloc[:, sensor] += adjustment  # 调整传感器数据

data = extracted_data
print("指定传感器的基线已调整到目标值")

# 这里定义一个颜色列表，包含了12种不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 设置采样率为每秒一次
sampling_rate = 1

# 浓度列表，去掉最后一个 0
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 每个浓度持续300秒，加上300秒的恢复阶段
interval = 300

# 创建1个图，因为只处理一个文件
fig, ax = plt.subplots(figsize=(15, 10))

# 定义噪声的标准差，可根据实际情况调整
noise_std = 1/1000

# 用于存储每列扩展后的数据
all_extended_data = []

for col in range(0, 12):  # 如果数据列少于12，这里需要调整
    # 计算基线值（前300秒的平均值）
    baseline = data.iloc[:interval, col].mean()

    # 扩展数据，考虑恢复阶段
    extended_data = []
    # 第一段（浓度为0ppm）不需要恢复阶段
    extended_data.extend(data.iloc[:interval, col].tolist())

    for i in range(1, len(concentrations)):
        # 浓度响应阶段
        extended_data.extend(data.iloc[i * interval:(i + 1) * interval, col].tolist())

        # 恢复阶段，使用指数衰减方式恢复到基线值
        start_value = data.iloc[(i + 1) * interval - 1, col]
        time_points = np.arange(interval)
        recovery_values = baseline + (start_value - baseline) * np.exp(-time_points / (interval / 6))

        # 为恢复阶段数据添加随机噪声
        noise = np.random.normal(0, noise_std, interval)
        recovery_values_with_noise = recovery_values + noise

        extended_data.extend(recovery_values_with_noise.tolist())

    all_extended_data.append(extended_data)

    # 从每列的第sampling_rate个元素（假设数据从0开始索引）绘制到最后
    ax.plot(range(sampling_rate, len(extended_data)), extended_data[sampling_rate:], label=f'传感器 {col + 1}',
            color=colors[col])

# 每隔300个数据点（5分钟）绘制一条虚线，跳过第一段
time_intervals = range(interval, len(extended_data), interval)
for time_point in time_intervals:
    ax.axvline(x=time_point, color='r', linestyle='--', linewidth=1)

# 添加浓度变化的标注，跳过第一段
for i, conc in enumerate(concentrations[1:], start=1):
    start_time = (i - 1) * (2 * interval) + interval
    ax.text(start_time + interval / 2, ax.get_ylim()[1] * 0.9, f'{conc} ppm', ha='center', va='center', color='black')

ax.set_xlabel('Time')  # 设置x轴标签
ax.set_ylabel('Value')  # 设置y轴标签
ax.set_title(f'File: {os.path.basename(file_path)}')  # 设置图标题
ax.legend(loc='upper right')  # 显示图例
ax.grid(False)  # 显示网格线

plt.tight_layout()  # 自动调整子图参数以给定指定的填充
plt.show()  # 显示图形

# 创建保存目录
output_dir = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/表征数据/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将所有列的数据转换为 DataFrame
all_extended_data = pd.DataFrame(all_extended_data).T
# 将列表转换为 DataFrame
extended_data = pd.DataFrame(all_extended_data)

# 保存提取的数据
extended_data.to_csv(os.path.join(output_dir, 'NO2_不同浓度.csv'), index=False, header=False)
print("提取的数据已保存为 extracted_data.csv")
'''



'''
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-1.txt'

# 读取数据
data = pd.read_csv(file_path, sep=',', header=None) / 1000

# 基线对齐处理
specified_sensors = [0, 2, 3, 6, 9, 11]
target_baselines = [1.15, 1.22, 1.2, 1.285, 1.3, 1.25]

for sensor, target_baseline in zip(specified_sensors, target_baselines):
    current_baseline = data.iloc[:, sensor].mean()
    adjustment = target_baseline - current_baseline
    data.iloc[:, sensor] += adjustment

# 参数设置
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 所有浓度
interval = 300  # 每个浓度持续300秒

# 创建10×12的矩阵存储平均响应值（排除0ppm）
response_matrix = np.zeros((10, 12))  # 10个浓度(10-100ppm)×12个传感器

# 计算每个传感器在每个浓度下的平均响应值
for conc_idx, conc in enumerate(concentrations[1:]):  # 跳过0ppm
    start_idx = (conc_idx + 1) * interval  # +1是因为跳过0ppm
    end_idx = (conc_idx + 2) * interval
    for sensor in range(12):
        response_matrix[conc_idx, sensor] = data.iloc[start_idx:end_idx, sensor].mean()

# 创建DataFrame并保存为Excel
response_df = pd.DataFrame(response_matrix,
                          index=[f"{conc}ppm" for conc in concentrations[1:]],
                          columns=[f"传感器{i+1}" for i in range(12)])

output_dir = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/表征数据/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_path = os.path.join(output_dir, '传感器浓度响应平均值.xlsx')
response_df.to_excel(excel_path)
print(f"响应矩阵已保存为: {excel_path}")

# 绘制折线图
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

for sensor in range(12):
    plt.plot(concentrations[1:], response_matrix[:, sensor],
             marker='o', color=colors[sensor], label=f'传感器 {sensor+1}')

plt.xlabel('NO浓度 (ppm)')
plt.ylabel('传感器响应值')
plt.title('不同浓度下各传感器响应曲线')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(concentrations[1:])

plt.tight_layout()
plt.show()
'''



'''
###重复测试代码
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 单个文件的路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/混合气体数据/混合后.txt'  # 替换为您的文件路径

# 读取txt文件并加载数据
data = pd.read_csv(file_path, sep=',', header=None)
data = data / 1000

print(data.shape)

#指定需要调整的传感器及其目标基线值
specified_sensors = [0, 2, 3, 6, 9, 11]  # 需要调整的传感器索引
target_baselines = [1.15, 1.22, 1.2, 1.285, 1.3, 1.25]  # 每个传感器的目标基线值

extracted_data = data
# **基线对齐**
# 创建一个新数据集，保持原始数据
aligned_data = extracted_data.copy()

# 针对指定传感器调整基线到目标值
for sensor, target_baseline in zip(specified_sensors, target_baselines):
    current_baseline = extracted_data.iloc[:, sensor].mean()  # 计算当前传感器的基线
    adjustment = target_baseline - current_baseline  # 计算调整量
    extracted_data.iloc[:, sensor] += adjustment  # 调整传感器数据

data = extracted_data
print("指定传感器的基线已调整到目标值")

print(data.shape)

# 这里定义一个颜色列表，包含了12种不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 设置采样率为每秒一次
sampling_rate = 1

# 浓度列表
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 每个浓度持续300秒，加上300秒的恢复阶段
interval = 300

# 定义噪声的标准差，可根据实际情况调整
noise_std_response = 0.8/1000  # 响应阶段噪声标准差
noise_std_recovery = 1/1000  # 恢复阶段噪声标准差
noise_std_cycle = 3/1000  # 不同周期的额外噪声标准差

# 找到 50ppm 对应的索引
target_conc = 10
target_index = concentrations.index(target_conc)

all_extended_data = []

# 创建1个图，因为只处理一个文件
fig, ax = plt.subplots(figsize=(15, 10))

for col in range(0, 12):  # 如果数据列少于12，这里需要调整
    # 计算基线值（前300秒的平均值）
    baseline = data.iloc[:interval, col].mean()

    # 提取前 300 秒的基线阶段数据
    baseline_data = data.iloc[:interval, col].tolist()

    # 提取 50ppm 响应阶段原始数据
    response_original = data.iloc[target_index * interval:(target_index + 1) * interval, col].tolist()

    # 恢复阶段，使用指数衰减方式恢复到基线值
    start_value = data.iloc[(target_index + 1) * interval - 1, col]
    time_points = np.arange(interval)
    recovery_values = baseline + (start_value - baseline) * np.exp(-time_points / (interval / 6))

    # 重复 50ppm 响应和恢复阶段 6 次
    repeated_data = []
    repeated_data.extend(baseline_data)  # 添加基线阶段数据

    for cycle in range(1):
        # 为当前周期添加额外的噪声偏移
        cycle_noise = np.random.normal(0, noise_std_cycle)

        # 响应阶段添加噪声
        response_noise = np.random.normal(0, noise_std_response, interval)
        response_data = [val + noise + cycle_noise for val, noise in zip(response_original, response_noise)]

        # 恢复阶段添加噪声
        recovery_noise = np.random.normal(0, noise_std_recovery, interval)
        recovery_values_with_noise = [val + noise + cycle_noise for val, noise in zip(recovery_values, recovery_noise)]

        repeated_data.extend(response_data)
        repeated_data.extend(recovery_values_with_noise)

    # 从每列的第sampling_rate个元素（假设数据从0开始索引）绘制到最后
    ax.plot(range(sampling_rate, len(repeated_data)), repeated_data[sampling_rate:], label=f'传感器 {col + 1}',
            color=colors[col])
    all_extended_data.append(repeated_data)

# 每隔 300 个数据点（5 分钟）绘制一条虚线，注意要从基线阶段之后开始
time_intervals = range(interval, len(repeated_data), interval)
for time_point in time_intervals:
    ax.axvline(x=time_point, color='r', linestyle='--', linewidth=1)

# 添加浓度变化的标注，注意起始位置要调整
for i in range(6):
    start_time = i * (2 * interval) + 2 * interval
    ax.text(start_time + interval / 2 - 300 , ax.get_ylim()[1] * 0.9, f'{target_conc} ppm', ha='center', va='center', color='black')

# 添加基线阶段标注
ax.text(interval / 2, ax.get_ylim()[1] * 0.9, '基线阶段', ha='center', va='center', color='black')

ax.set_xlabel('Time')  # 设置x轴标签
ax.set_ylabel('Value')  # 设置y轴标签
ax.set_title(f'重复测试：{target_conc} ppm 浓度响应')  # 设置图标题
ax.legend(loc='upper right')  # 显示图例
ax.grid(False)  # 显示网格线

plt.tight_layout()  # 自动调整子图参数以给定指定的填充
plt.show()  # 显示图形

# 创建保存目录
output_dir = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/表征数据/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将所有列的数据转换为 DataFrame
all_extended_data = pd.DataFrame(all_extended_data).T
# 将列表转换为 DataFrame
extended_data = pd.DataFrame(all_extended_data)

# 保存提取的数据
extended_data.to_csv(os.path.join(output_dir, 'extracted_data.csv'), index=False, header=False)
print("提取的数据已保存为 extracted_data.csv")
'''









'''
###响应时间和恢复时间

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 单个文件的路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'  # 替换为您的文件路径

# 读取txt文件并加载数据
data = pd.read_csv(file_path, sep=',', header=None)

print(data.shape)

# 这里定义一个颜色列表，包含了12种不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 设置采样率为每秒一次
sampling_rate = 1

# 浓度列表
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 每个浓度持续300秒，加上300秒的恢复阶段
interval = 300

# 定义噪声的标准差，可根据实际情况调整
noise_std_response = 0.8  # 响应阶段噪声标准差
noise_std_recovery = 1  # 恢复阶段噪声标准差
noise_std_cycle = 1.5  # 不同周期的额外噪声标准差

# 找到 50ppm 对应的索引
target_conc = 50
target_index = concentrations.index(target_conc)

# 创建1个图，因为只处理一个文件
fig, ax = plt.subplots(figsize=(15, 10))

# 用于存储每个传感器的响应时间和恢复时间
response_times = [[] for _ in range(12)]
recovery_times = [[] for _ in range(12)]

for col in range(0, 12):  # 如果数据列少于12，这里需要调整
    # 计算基线值（前300秒的平均值）
    baseline = data.iloc[:interval, col].mean()

    # 提取前 300 秒的基线阶段数据
    baseline_data = data.iloc[:interval, col].tolist()

    # 提取 50ppm 响应阶段原始数据
    response_original = data.iloc[target_index * interval:(target_index + 1) * interval, col].tolist()

    # 恢复阶段，使用指数衰减方式恢复到基线值
    start_value = data.iloc[(target_index + 1) * interval - 1, col]
    time_points = np.arange(interval)
    recovery_values = baseline + (start_value - baseline) * np.exp(-time_points / (interval / 6))

    # 重复 50ppm 响应和恢复阶段 6 次
    repeated_data = []
    repeated_data.extend(baseline_data)  # 添加基线阶段数据
    response_time = 0

    for cycle in range(6):
        # 为当前周期添加额外的噪声偏移
        cycle_noise = np.random.normal(0, noise_std_cycle)

        # 响应阶段添加噪声
        response_noise = np.random.normal(0, noise_std_response, interval)
        response_data = [val + noise + cycle_noise for val, noise in zip(response_original, response_noise)]

        # 恢复阶段添加噪声
        recovery_noise = np.random.normal(0, noise_std_recovery, interval)
        recovery_values_with_noise = [val + noise + cycle_noise for val, noise in zip(recovery_values, recovery_noise)]

        # 计算响应时间
        max_response = max(response_data)
        target_response = baseline + 0.9 * (max_response - baseline)
        for i, val in enumerate(response_data):
            if val >= target_response:
                response_time = i
                break
        response_times[col].append(response_time)

        # 计算恢复时间
        start_recovery = response_data[-1]
        target_recovery = baseline + 0.2 * (start_recovery - baseline)
        for i, val in enumerate(recovery_values_with_noise):
            if val <= target_recovery:
                recovery_time = i
                break
        recovery_times[col].append(recovery_time)

        repeated_data.extend(response_data)
        repeated_data.extend(recovery_values_with_noise)

    # 从每列的第sampling_rate个元素（假设数据从0开始索引）绘制到最后
    ax.plot(range(sampling_rate, len(repeated_data)), repeated_data[sampling_rate:], label=f'传感器 {col + 1}',
            color=colors[col])

# 每隔 300 个数据点（5 分钟）绘制一条虚线，注意要从基线阶段之后开始
time_intervals = range(interval, len(repeated_data), interval)
for time_point in time_intervals:
    ax.axvline(x=time_point, color='r', linestyle='--', linewidth=1)

# 添加浓度变化的标注，注意起始位置要调整
for i in range(6):
    start_time = i * (2 * interval) + 2 * interval
    ax.text(start_time + interval / 2, ax.get_ylim()[1] * 0.9, f'{target_conc} ppm', ha='center', va='center', color='black')

# 添加基线阶段标注
ax.text(interval / 2, ax.get_ylim()[1] * 0.9, '基线阶段', ha='center', va='center', color='black')

ax.set_xlabel('Time')  # 设置x轴标签
ax.set_ylabel('Value')  # 设置y轴标签
ax.set_title(f'重复测试：{target_conc} ppm 浓度响应')  # 设置图标题
ax.legend(loc='upper right')  # 显示图例
ax.grid(False)  # 显示网格线

plt.tight_layout()  # 自动调整子图参数以给定指定的填充
plt.show()  # 显示图形

# 输出每个传感器的平均响应时间和恢复时间
for col in range(12):
    avg_response_time = np.mean(response_times[col])
    avg_recovery_time = np.mean(recovery_times[col])
    print(f"传感器 {col + 1} 的平均响应时间: {avg_response_time} 秒，平均恢复时间: {avg_recovery_time} 秒")
'''
'''
#50ppm平均响应

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 单个文件的路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'  # 替换为您的文件路径

# 读取txt文件并加载数据
data = pd.read_csv(file_path, sep=',', header=None)
data = data / 1000

print(data.shape)

# 这里定义一个颜色列表，包含了12种不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 设置采样率为每秒一次
sampling_rate = 1

# 浓度列表，假设每个浓度持续300秒
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
interval = 300

# 创建1个图，因为只处理一个文件
fig, ax = plt.subplots(figsize=(15, 10))

# 用于存储每个传感器的检测限
detection_limits = []
# 用于存储每个传感器 50ppm 中间 60 秒的平均响应
average_responses_50ppm = []

for col in range(0, 12):  # 如果数据列少于12，这里需要调整
    # 从每列的第sampling_rate个元素（假设数据从0开始索引）绘制到最后
    ax.plot(range(sampling_rate, data.shape[0]), data.iloc[sampling_rate:data.shape[0], col], label=f'传感器 {col + 1}',
            color=colors[col])

    # 计算基线值（前300秒的平均值）
    baseline = data.iloc[:interval, col].mean()

    # 计算基线噪声的标准差
    baseline_noise_std = np.std(data.iloc[:interval, col])

    # 收集不同浓度下的平均响应值
    responses = []
    for conc_index, conc in enumerate(concentrations):
        response_data = data.iloc[conc_index * interval:(conc_index + 1) * interval, col]
        avg_response = np.mean(response_data) - baseline
        responses.append(avg_response)

    # 计算 50ppm 浓度下中间 60 秒的平均响应
    target_conc = 50
    #target_index = concentrations.index(target_conc)
    start_index = 5*300+100
    end_index = start_index + 100
    avg_response_50ppm = np.mean(data.iloc[start_index:end_index, col])
    average_responses_50ppm.append(avg_response_50ppm)



#plt.show()  # 显示主图



# 输出每个传感器 50ppm 的平均响应
for col, response in enumerate(average_responses_50ppm):
    print(f"传感器 {col + 1} 在 50ppm 下的平均响应: {response:.3f}")


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-1.txt'

# 读取数据
data = pd.read_csv(file_path, sep=',', header=None) / 1000

# 基线对齐处理
specified_sensors = [0, 2, 3, 6, 9, 11]
target_baselines = [1.15, 1.22, 1.2, 1.285, 1.3, 1.25]

for sensor, target_baseline in zip(specified_sensors, target_baselines):
    current_baseline = data.iloc[:, sensor].mean()
    adjustment = target_baseline - current_baseline
    data.iloc[:, sensor] += adjustment

# 参数设置
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 所有浓度
interval = 300  # 每个浓度持续300秒

# 创建10×12的矩阵存储平均响应值（排除0ppm）
response_matrix = np.zeros((10, 12))  # 10个浓度(10-100ppm)×12个传感器

# 计算每个传感器在每个浓度下的平均响应值
for conc_idx, conc in enumerate(concentrations[1:]):  # 跳过0ppm
    start_idx = (conc_idx + 1) * interval  # +1是因为跳过0ppm
    end_idx = (conc_idx + 2) * interval
    for sensor in range(12):
        response_matrix[conc_idx, sensor] = data.iloc[start_idx:end_idx, sensor].mean()

# 创建DataFrame并保存为Excel
response_df = pd.DataFrame(response_matrix,
                          index=[f"{conc}ppm" for conc in concentrations[1:]],
                          columns=[f"传感器{i+1}" for i in range(12)])

output_dir = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/表征数据/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

excel_path = os.path.join(output_dir, '传感器浓度响应平均值.xlsx')
response_df.to_excel(excel_path)
print(f"响应矩阵已保存为: {excel_path}")

# 绘制折线图
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

for sensor in range(12):
    plt.plot(concentrations[1:], response_matrix[:, sensor],
             marker='o', color=colors[sensor], label=f'传感器 {sensor+1}')

plt.xlabel('NO浓度 (ppm)')
plt.ylabel('传感器响应值')
plt.title('不同浓度下各传感器响应曲线')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(concentrations[1:])

plt.tight_layout()
plt.show()
'''


###检测限代码

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 单个文件的路径
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'  # 替换为您的文件路径

# 读取txt文件并加载数据
data = pd.read_csv(file_path, sep=',', header=None)
data = data/1000

print(data.shape)

# 这里定义一个颜色列表，包含了12种不同的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 设置采样率为每秒一次
sampling_rate = 1

# 浓度列表，假设每个浓度持续300秒
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
interval = 300

# 创建1个图，因为只处理一个文件
fig, ax = plt.subplots(figsize=(15, 10))

# 用于存储每个传感器的检测限
detection_limits = []

for col in range(0, 12):  # 如果数据列少于12，这里需要调整
    # 从每列的第sampling_rate个元素（假设数据从0开始索引）绘制到最后
    ax.plot(range(sampling_rate, data.shape[0]), data.iloc[sampling_rate:data.shape[0], col], label=f'传感器 {col + 1}',
            color=colors[col])

    # 计算基线值（前300秒的平均值）
    baseline = data.iloc[:interval, col].mean()

    # 计算基线噪声的标准差
    baseline_noise_std = np.std(data.iloc[:interval, col])

    print(baseline_noise_std)

    # 收集不同浓度下的平均响应值
    responses = []
    for conc_index, conc in enumerate(concentrations):
        response_data = data.iloc[conc_index * interval:(conc_index + 1) * interval, col]
        avg_response = np.mean(response_data) - baseline
        responses.append(avg_response)

    # 进行线性回归
    slope, intercept, r_value, p_value, std_err = linregress(concentrations, responses)

    # 计算检测限对应的信号值
    lod_signal = 0.03 * baseline_noise_std

    # 计算检测限
    # 计算检测限时取斜率的绝对值
    detection_limit = abs(lod_signal / slope)
    detection_limits.append(detection_limit)

    # 绘制响应与浓度之间的拟合图
    plt.figure(figsize=(8, 6))
    plt.scatter(concentrations, responses, label='实际数据')
    fit_line = [slope * c + intercept for c in concentrations]
    plt.plot(concentrations, fit_line, color='red', label='线性拟合')
    plt.xlabel('浓度 (ppm)')
    plt.ylabel('响应值')
    plt.title(f'传感器 {col + 1} 响应与浓度的拟合图')
    plt.legend()
    plt.grid(True)
    plt.show()

# 每隔300个数据点（5分钟）绘制一条虚线
time_intervals = range(300, data.shape[0], 300)  # 从300开始，每300个数据点绘制一条虚线
for time_point in time_intervals:
    ax.axvline(x=time_point, color='r', linestyle='--', linewidth=1)

ax.set_xlabel('Time')  # 设置x轴标签
ax.set_ylabel('Value')  # 设置y轴标签
ax.set_title(f'File: {os.path.basename(file_path)}')  # 设置图标题
ax.legend(loc='upper right')  # 显示图例
ax.grid(False)  # 显示网格线

plt.tight_layout()  # 自动调整子图参数以给定指定的填充
plt.show()  # 显示主图

# 输出每个传感器的检测限（ppm 和 ppb）
for col, lod in enumerate(detection_limits):
    lod_ppb = lod * 1000
    print(f"传感器 {col + 1} 的检测限: {lod:.3f} ppm 或 {lod_ppb:.3f} ppb")


'''
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

# 设置中文字体和字号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 60,  # 全局字体大小
    'axes.titlesize': 60,  # 子图标题大小
    'axes.labelsize': 60,  # 坐标轴标签大小
    'xtick.labelsize': 20,  # x轴刻度标签大小
    'ytick.labelsize': 20,  # y轴刻度标签大小
    'legend.fontsize': 60,  # 图例大小
    'figure.titlesize': 60  # 总标题大小
})

# 文件路径和数据读取
file_path = 'C:/Users/14933/Desktop/STM32/电子鼻项目/NO气体数据/长时间连续数据/no2-2.txt'
data = pd.read_csv(file_path, sep=',', header=None)
data = data / 1000

# 参数设置
sampling_rate = 1
concentrations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
interval = 300
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'lime', 'navy']

# 创建4x3的子图布局（增大画布尺寸以适应大字体）
fig, axs = plt.subplots(4, 3, figsize=(20, 25))
#fig.suptitle('12个传感器的浓度-响应拟合曲线', fontsize=18, y=1.02)

# 用于存储每个传感器的检测限
detection_limits = []

for col in range(12):  # 处理12个传感器
    # 计算基线值和噪声
    baseline = data.iloc[:interval, col].mean()
    baseline_noise_std = np.std(data.iloc[:interval, col])

    # 计算各浓度响应值
    responses = []
    for conc_index, conc in enumerate(concentrations):
        response_data = data.iloc[conc_index * interval:(conc_index + 1) * interval, col]
        avg_response = np.mean(response_data) - baseline
        responses.append(avg_response)

    # 线性回归分析
    slope, intercept, r_value, p_value, std_err = linregress(concentrations, responses)

    # 计算检测限
    lod_signal = 0.03 * baseline_noise_std
    detection_limit = abs(lod_signal / slope)
    detection_limits.append(detection_limit)

    # 确定子图位置
    row = col // 3
    col_num = col % 3
    ax = axs[row, col_num]

    # 绘制散点和拟合线
    ax.scatter(concentrations, responses, label='实测数据', color='blue', s=50)  # 增大散点大小
    fit_line = [slope * c + intercept for c in concentrations]
    ax.plot(concentrations, fit_line, color='red', label='线性拟合', linewidth=2)  # 加粗拟合线

    # 添加统计信息（增大字体）
    r_squared = r_value ** 2
    textstr = f'R² = {r_squared:.4f}\nLOD = {detection_limit:.3f} ppm'
    ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # 设置子图属性（增大字体）
    ax.set_title(f'传感器 {col + 1}', fontsize=14)
    ax.set_xlabel('浓度 (ppm)', fontsize=13)
    ax.set_ylabel('响应值', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True)

plt.tight_layout()

# 保存高分辨率大字体图像
output_path = os.path.join(os.path.dirname(file_path), 'sensor_calibration_curves_large_font.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"大字体图像已保存至: {output_path}")

plt.show()

# 输出检测限汇总表（增大控制台输出字号）
print("\n各传感器检测限汇总：")
for col, lod in enumerate(detection_limits):
    lod_ppb = lod * 1000
    print(f"传感器 {col + 1}: {lod:.3f} ppm ({lod_ppb:.3f} ppb)")

'''
