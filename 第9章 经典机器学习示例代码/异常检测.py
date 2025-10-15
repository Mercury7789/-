import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 生成模拟数据（95%正常数据，5%异常数据）
np.random.seed(42)
normal_data = 0.5 * np.random.randn(500, 2)  # 正常数据
outliers = np.random.uniform(low=-4, high=4, size=(25, 2))  # 异常数据
X = np.vstack([normal_data, outliers])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练异常检测模型
clf = IsolationForest(contamination=0.05, random_state=42)
y_pred = clf.fit_predict(X_scaled)  # 1表示正常，-1表示异常

# 输出结果
n_outliers = (y_pred == -1).sum()
print(f"检测出的异常点数量: {n_outliers}")
print("异常点索引:", np.where(y_pred == -1)[0])

import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制正常点
plt.scatter(X_scaled[y_pred == 1, 0], X_scaled[y_pred == 1, 1],
            c='blue', edgecolor='k', s=20, label='正常点')

# 绘制异常点
plt.scatter(X_scaled[y_pred == -1, 0], X_scaled[y_pred == -1, 1],
            c='red', edgecolor='k', s=50, label='异常点')

# 添加标题和标签
plt.title('异常检测结果可视化', fontsize=14)
plt.xlabel('特征1（标准化后）', fontsize=12)
plt.ylabel('特征2（标准化后）', fontsize=12)
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()