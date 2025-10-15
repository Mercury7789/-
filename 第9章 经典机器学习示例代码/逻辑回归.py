# 导入必要的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 加载示例数据集 - 乳腺癌数据集
cancer = datasets.load_breast_cancer()
X = cancer.data[:, :2]  # 只取前两个特征以便可视化
y = cancer.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建逻辑回归分类器
lr_classifier = LogisticRegression(
    penalty='l2',  # 正则化类型(L2正则化)
    C=1.0,         # 正则化强度的倒数(越小正则化越强)
    solver='lbfgs',# 优化算法
    max_iter=100   # 最大迭代次数
)

# 训练模型
lr_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = lr_classifier.predict(X_test)
y_prob = lr_classifier.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

# 评估模型
print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化决策边界和概率
plt.figure(figsize=(12, 5))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建网格来绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点的概率
Z = lr_classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# 绘制决策边界和概率分布
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.colorbar()
plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])
plt.title('逻辑回归概率分布')

# 绘制决策边界(0.5阈值)
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z > 0.5, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])
plt.title('决策边界(阈值=0.5)')

plt.tight_layout()
plt.show()