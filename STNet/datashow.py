import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 根据你的数据调整
# 这里创建一个9x9的二维矩阵，表示不同浓度组合的热力图
# 假设你已经知道每个NO浓度与NO2浓度的对应关系

data1 = [
    [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10的NO2浓度
    [90, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20的NO2浓度
    [80, 90, 100, 0, 0, 0, 0, 0, 0, 0, 0],  # 30的NO2浓度
    [70, 80, 90,100, 0, 0, 0, 0, 0, 0, 0],  # 40的NO2浓度
    [60, 70, 80, 90, 100, 0, 0, 0, 0, 0, 0],  # 50的NO2浓度
    [50, 60, 70, 80, 90, 100, 0, 0, 0, 0, 0],  # 60的NO2浓度
    [40, 50, 60, 70, 80, 0, 0, 0, 0, 0, 0],  # 70的NO2浓度
    [30, 40, 50, 60, 0, 0, 0, 0, 0, 0, 0],  # 80的NO2浓度
    [20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0],   # 90的NO2浓度
    [10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 90的NO2浓度
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # 90的NO2浓度
]

data = [
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # 10的NO2浓度
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0],  # 20的NO2浓度
    [20, 0, 40, 50, 60, 70, 80, 90, 100, 0, 0],  # 30的NO2浓度
    [30, 0, 0,60, 70, 80, 90, 100, 0, 0, 0],  # 40的NO2浓度
    [40, 0, 0, 0, 80, 90, 100, 0, 0, 0, 0],  # 50的NO2浓度
    [50, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0],  # 60的NO2浓度
    [60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 70的NO2浓度
    [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 80的NO2浓度
    [80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 90的NO2浓度
    [90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 90的NO2浓度
    [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 90的NO2浓度
]


# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22  # Base font size

# Set figure size
plt.figure(figsize=(8, 11))  # Adjusted to 8 inches height for better proportions

# Create heatmap with integer display
ax = sns.heatmap(data, cmap="YlGnBu", annot=True, fmt='d',
                xticklabels=[f"{i}" for i in range(0, 102, 10)],
                yticklabels=[f"{i}" for i in range(0, 102, 10)],
                cbar_kws={'label': 'Concentration Ratio'},
                annot_kws={'size': 16})  # Annotation font size

# Set title and axis labels with custom font sizes
ax.set_xlabel(r"NO Concentration (ppm)", fontsize=28, fontweight='bold')
ax.set_ylabel("NO$_2$ Concentration (ppm)", fontsize=28, fontweight='bold')
#ax.set_title("Heatmap of NO and NO$_2$ Concentration Ratio", fontsize=28, fontweight='bold')

# Adjust tick label sizes
ax.tick_params(axis='both', which='major', labelsize=18)

# Adjust colorbar label size
cbar = ax.collections[0].colorbar
cbar.set_label('Concentration Ratio', size=20, weight='bold')

plt.tight_layout()  # Prevent label cutoff
plt.show()
