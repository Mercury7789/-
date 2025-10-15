import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据准备
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 2. 超简CNN模型 (仅4层)
model = nn.Sequential(
    nn.Conv2d(1, 6, 3),    # 1输入通道,6输出通道,3x3卷积核
    nn.ReLU(),
    nn.MaxPool2d(2),       # 2x2最大池化
    nn.Flatten(),          # 展平
    nn.Linear(6*13*13, 10) # 全连接输出层 (28x28 -> 卷积后13x13)
)

# 3. 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 4. 极简训练循环
for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}完成')

# 5. 可视化预测
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
model.eval()
sample = next(iter(train_loader))[0][:5]
with torch.no_grad():
    preds = model(sample).argmax(dim=1)

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(sample[i][0], cmap='gray')
    plt.title(f'预测: {preds[i].item()}')
    plt.axis('off')
plt.show()