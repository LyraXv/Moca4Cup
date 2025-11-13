# 设置样式
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

# plt.style.use('seaborn-v0_8')
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel('../eval_res/res_paraK.xlsx')

# 定义颜色和标记
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 更多颜色备用
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
line_styles = ['-', '--', '-.', ':']

# 创建单张图
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='white')
ax.set_facecolor('white')  # 设置坐标轴区域背景为白色

# 选择一个方法（这里选择第一个方法，你可以根据需要修改）
methods = df['Method'].unique()
selected_method = methods[0]  # 选择第一个方法，你可以改为其他方法

# 筛选该方法的数揗
method_data = df[df['Method'] == selected_method]
Levels = sorted(method_data['Level'].unique())
# 在绘制折线的循环中，修改标签部分
title_list = ['Easy', 'Normal', 'Complex']
for i, Level in enumerate(Levels):
    level_data = method_data[method_data['Level'] == Level].sort_values('Sample K')

    # 为每个等级分配独特的样式
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    line_style = line_styles[i % len(line_styles)]

    # 绘制折线 - 使用更有意义的标签
    label_name = title_list[i] if i < len(title_list) else f'Level {Level}'
    ax.plot(level_data['Sample K'], level_data['Accuracy'],
            color=color, marker=marker, linestyle=line_style,
            linewidth=2.5, markersize=8,
            label=label_name,
            markeredgecolor='white', markeredgewidth=1)


# 美化图表
# 设置黑色坐标轴和增大字号
ax.set_xlabel('Sample Size(K)', fontsize=16, fontweight='bold', color='black')  # 增大字号，黑色
ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold', color='black')  # 增大字号，黑色

# 设置坐标轴颜色和标签颜色 - 特别调整刻度数字大小
ax.tick_params(axis='both', which='major', labelsize=12, colors='black')  # labelsize控制刻度数字大小
# ax.set_xlabel('Sample Size(K)', fontsize=16, fontweight='bold')
# ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
# ax.set_title(f'Accuracy vs Sample K - {selected_method}', fontsize=18, fontweight='bold')
ax.legend(fontsize=14, loc='best')
ax.grid(True, alpha=0.3)

# 设置横坐标范围
ax.set_xlim(-0.5, 15.5)

# 设置指定的横坐标刻度
custom_xticks = [0, 1, 3, 5, 7, 9, 11, 13, 15]
ax.set_xticks(custom_xticks)

# 设置纵坐标范围
accuracy_data = method_data['Accuracy']

# 自动判断最佳显示范围
if accuracy_data.max() - accuracy_data.min() < 0.1:
    # 数据差异很小，大幅放大
    margin = (accuracy_data.max() - accuracy_data.min()) * 0.5
    y_min = max(0, accuracy_data.min() - margin)
    y_max = min(1, accuracy_data.max() + margin)
elif accuracy_data.max() - accuracy_data.min() < 0.3:
    # 数据差异中等，适度放大
    margin = (accuracy_data.max() - accuracy_data.min()) * 0.3
    y_min = max(0, accuracy_data.min() - margin)
    y_max = min(1, accuracy_data.max() + margin)
else:
    # 数据差异大，使用完整范围
    y_min = 0.05
    y_max = 0.55
y_min = 0.05
y_max = 0.55

ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 1.0 = 100%

# 调整布局
plt.tight_layout(pad=3.0)
plt.savefig('single_method_all_levels_highres.png', dpi=300, bbox_inches='tight')
plt.show()
