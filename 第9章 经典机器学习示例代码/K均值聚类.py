# 导入必要的库
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 生成模拟数据（300个样本，3个簇）
X, y_true = make_blobs(n_samples=300, centers=3,
                       cluster_std=0.8, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建并训练K均值模型
kmeans = KMeans(
    n_clusters=3,      # 簇数量
    init='k-means++',  # 初始化方法
    max_iter=300,      # 最大迭代次数
    random_state=42
).fit(X_scaled)

# 获取结果
y_pred = kmeans.labels_          # 预测的簇标签
centroids = kmeans.cluster_centers_  # 聚类中心坐标

# 评估指标
inertia = kmeans.inertia_        # 簇内平方和
silhouette_avg = silhouette_score(X_scaled, y_pred)  # 轮廓系数

# 打印结果
print(f"簇内平方和(SSE): {inertia:.2f}")
print(f"轮廓系数: {silhouette_avg:.3f}")
print("\n聚类中心坐标(标准化后):")
print(centroids)
print("\n聚类中心坐标(原始尺度):")
print(scaler.inverse_transform(centroids))


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置图形样式
plt.figure(figsize=(12, 5))

# 子图1：原始数据分布
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.7)
plt.title("原始数据分布 (真实标签)")
plt.xlabel("特征1")
plt.ylabel("特征2")
plt.grid(True, linestyle='--', alpha=0.5)

# 子图2：K均值聚类结果
plt.subplot(1, 2, 2)
# 绘制聚类结果
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30, alpha=0.7)
# 标记聚类中心
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='red', marker='X', s=200, alpha=0.9, linewidths=2,
            edgecolors='black', label='聚类中心')
plt.title(f"K均值聚类结果 (K=3)\n轮廓系数: {silhouette_avg:.3f}")
plt.xlabel("特征1")
plt.ylabel("特征2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()