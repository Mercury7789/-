import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建正弦波序列数据
x = np.linspace(0, 50, 500)
y = np.sin(x)


# 2. 准备训练数据
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - 1):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)


seq_length = 10
X, y = create_sequences(y, seq_length)
X = X.unsqueeze(2)  # 添加特征维度 (样本数, 序列长度, 特征数)


# 3. 定义简单RNN模型
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # out包含所有时间步的输出
        return self.fc(out[:, -1, :])  # 只取最后一个时间步的输出


model = SimpleRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. 预测并可视化
with torch.no_grad():
    predictions = model(X)

plt.figure(figsize=(10, 5))
plt.plot(y.numpy(), label='True Values')
plt.plot(predictions.numpy(), label='Predictions')
plt.legend()
plt.show()