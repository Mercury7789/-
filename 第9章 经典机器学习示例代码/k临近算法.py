# 导入必要的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 加载示例数据集 - 鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 只取前两个特征以便可视化
y = iris.target

# 数据标准化（KNN对特征尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

# 创建KNN分类器
knn_classifier = KNeighborsClassifier(
    n_neighbors=5,  # K值
    weights='uniform',  # 权重方式：uniform(平均)或distance(加权)
    metric='euclidean'  # 距离度量：欧氏距离
)

# 训练模型
knn_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = knn_classifier.predict(X_test)

# 评估模型
print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化决策边界
plt.figure(figsize=(10, 6))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建网格
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点类别
Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.4)
# 绘制训练样本
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                     s=30, edgecolor='k', cmap=plt.cm.Set1)
# 绘制测试样本（用X标记）
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            s=100, marker='x', edgecolor='k', cmap=plt.cm.Set1)

# 添加图例和标签
plt.xlabel("标准化后的花萼长度")
plt.ylabel("标准化后的花萼宽度")
plt.title("KNN分类结果 (K=5)")
plt.legend(*scatter.legend_elements(), title="类别")
plt.show()