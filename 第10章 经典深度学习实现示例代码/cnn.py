import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 1. 定义超简单CNN模型
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 保持尺寸不变
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)  # MNIST 10类输出

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [64, 16, 14, 14]
        return self.fc(x.view(-1, 16 * 14 * 14))


# 2. 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# 3. 初始化模型和优化器
model = TinyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
print("开始训练...")
for epoch in range(3):  # 只训练3个epoch
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# 5. 测试评估
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'测试准确率: {100 * correct / total:.2f}%')

# 6. 可视化第一个卷积核的效果
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sample_img = next(iter(train_loader))[0][0]  # 取第一个训练样本
conv1_weight = model.conv1.weight.data.numpy()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title('输入图像')

plt.subplot(1,3,2)
plt.imshow(conv1_weight[0,0], cmap='gray')
plt.title('卷积核1')

plt.subplot(1,3,3)
with torch.no_grad():
    conv_output = model.conv1(sample_img.unsqueeze(0))
plt.imshow(conv_output[0,0].numpy(), cmap='gray')
plt.title('卷积输出')
plt.show()