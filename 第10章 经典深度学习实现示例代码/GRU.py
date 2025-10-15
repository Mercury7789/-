import torch
import torch.nn as nn
import numpy as np

# 1. 数据准备
text = "hello gru"  # 训练文本
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
encoded = [char_to_idx[c] for c in text]

# 2. 超参数设置
input_size = len(chars)  # 输入维度（字符表大小）
hidden_size = 32  # 隐藏层维度
seq_length = 3  # 序列长度
batch_size = 1  # 批大小


# 3. 定义GRU模型（比LSTM更简单）
class TinyGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden


# 4. 初始化
model = TinyGRU()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 训练循环
for epoch in range(100):
    # 初始化隐藏状态（GRU只需一个hidden state）
    hidden = torch.zeros(1, batch_size, hidden_size)

    # 准备输入输出
    start_idx = np.random.randint(0, len(encoded) - seq_length)
    input_seq = encoded[start_idx:start_idx + seq_length]
    target_seq = encoded[start_idx + 1:start_idx + seq_length + 1]

    # 转换为one-hot
    inputs = torch.zeros(batch_size, seq_length, input_size)
    for i in range(seq_length):
        inputs[0][i][input_seq[i]] = 1

    # 训练步骤
    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs.squeeze(0), torch.LongTensor(target_seq))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


# 6. 文本生成
def generate(start_char, length=10):
    with torch.no_grad():
        hidden = torch.zeros(1, 1, hidden_size)
        input_char = torch.zeros(1, 1, input_size)
        input_char[0][0][char_to_idx[start_char]] = 1

        output_text = start_char
        for _ in range(length):
            output, hidden = model(input_char, hidden)
            char_idx = torch.argmax(output[0, -1]).item()
            output_text += idx_to_char[char_idx]
            input_char = torch.zeros(1, 1, input_size)
            input_char[0][0][char_idx] = 1

        return output_text


print("\n生成结果:", generate('h'))