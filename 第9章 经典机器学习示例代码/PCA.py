# 导入必要的库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建PCA模型
pca = PCA(n_components=2)  # 降为2维

# 执行PCA变换
X_pca = pca.fit_transform(X_scaled)

# 输出结果
print("原始数据形状:", X.shape)
print("降维后数据形状:", X_pca.shape)
print("\n主成分方向(特征向量):")
print(pca.components_)
print("\n各主成分的方差解释比例:")
print(pca.explained_variance_ratio_)
print("\n累计方差解释比例:", sum(pca.explained_variance_ratio_))



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制PCA结果（按类别着色）
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                label=name, edgecolor='k', s=60)

# 添加标签和标题
plt.xlabel('第一主成分 (解释方差: {:.1%})'.format(pca.explained_variance_ratio_[0]))
plt.ylabel('第二主成分 (解释方差: {:.1%})'.format(pca.explained_variance_ratio_[1]))
plt.title('鸢尾花数据集PCA降维结果')

# 添加图例
plt.legend(title="鸢尾花种类")

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()