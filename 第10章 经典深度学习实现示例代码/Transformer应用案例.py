import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据 - 简单正弦波
t = np.linspace(0, 10, 500)
data = np.sin(t)

# 2. 准备数据
seq_len = 20
X = torch.FloatTensor(np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)]))
y = torch.FloatTensor(np.array([data[i + seq_len] for i in range(len(data) - seq_len)]))
X = X.unsqueeze(2)  # (480, 20, 1)
y = y.unsqueeze(1)  # (480, 1)


# 3. 修正后的极简Transformer模型
class FixedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encoder = nn.Linear(1, 16)  # 增加特征维度到16
        self.transformer = nn.Transformer(
            d_model=16,  # 匹配pos_encoder的输出
            nhead=2,  # 注意力头数
            num_encoder_layers=1,
            num_decoder_layers=0,
            dim_feedforward=32
        )
        self.fc = nn.Linear(16, 1)  # 输出层

    def forward(self, x):
        # 输入x形状: (batch, seq, 1)
        x = self.pos_encoder(x)  # (batch, seq, 16)

        # Transformer需要(seq, batch, features)
        x = x.permute(1, 0, 2)  # (seq, batch, 16)

        # Transformer处理 (使用x同时作为src和tgt)
        out = self.transformer(x, x)  # (seq, batch, 16)

        # 取最后一个时间步
        out = out[-1]  # (batch, 16)

        # 最终预测
        return self.fc(out)  # (batch, 1)


model = FixedTransformer()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 4. 训练循环
losses = []
for epoch in range(100):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y.numpy(), label='True')
plt.plot(model(X).detach().numpy(), '--', label='Predicted')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('Training Loss')
plt.show()