# 导入必要的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 加载示例数据集 - 鸢尾花数据集(只取两类)
iris = datasets.load_iris()
X = iris.data[:100, :2]  # 只取前两个特征和前100个样本(两类)
y = iris.target[:100]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建SVM分类器 - 使用线性核
svm_classifier = SVC(kernel='linear', C=1.0)

# 训练模型
svm_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出结果
print("测试集准确率: {:.2f}%".format(accuracy * 100))
print("支持向量数量:", len(svm_classifier.support_vectors_))
print("支持向量索引:", svm_classifier.support_)

# 创建网格来绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点的类别
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 在绘图代码前添加以下设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 然后是你原来的绘图代码
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.scatter(svm_classifier.support_vectors_[:, 0],
            svm_classifier.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k')
plt.title('SVM决策边界与支持向量')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.show()