import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟温度数据（昼夜温度变化）
days = 30
hours = days * 24
t = np.linspace(0, days, hours)
data = 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, hours) + 20


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


seq_length = 24  # 使用24小时数据预测下一小时温度
X, y = create_sequences(data, seq_length)
X = X.unsqueeze(2)  # (样本数, 序列长度, 特征数=1)


# 3. 定义简单GRU模型
class TempPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


model = TempPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
for epoch in range(50):
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

plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Temp')
plt.plot(range(seq_length, len(predictions) + seq_length), predictions.numpy(),
         label='Predicted Temp', alpha=0.7)
plt.title('Temperature Prediction with GRU')
plt.legend()
plt.show()