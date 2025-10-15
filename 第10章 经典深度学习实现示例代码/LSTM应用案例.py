import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1. 生成模拟销售数据
def generate_sales_data(length=200):
    # 基础趋势 + 季节性 + 随机噪声
    x = np.linspace(0, 10, length)
    trend = 0.5 * x
    seasonal = 5 * np.sin(2 * np.pi * x / 25)
    noise = np.random.normal(0, 1, length)
    return trend + seasonal + noise


data = generate_sales_data()


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


seq_length = 15
X, y = create_sequences(data, seq_length)
X = X.unsqueeze(2)  # 添加特征维度 (样本数, 序列长度, 特征数)


# 3. 定义简单LSTM模型
class SalesPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=20, batch_first=True)
        self.linear = nn.Linear(20, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


model = SalesPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. 可视化结果
with torch.no_grad():
    predictions = model(X)

plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Sales')
plt.plot(range(seq_length, len(predictions) + seq_length), predictions.numpy(),
         label='Predicted Sales', linestyle='--')
plt.title('Sales Prediction with LSTM')
plt.legend()
plt.show()