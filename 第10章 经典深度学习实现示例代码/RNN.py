import torch
import torch.nn as nn
import numpy as np

# 1. 准备数据
text = "hello pytorch"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
encoded = [char_to_idx[c] for c in text]

# 2. 参数设置
input_size = len(chars)  # 输入特征维度
hidden_size = 32  # 隐藏层维度
seq_length = 5  # 序列长度
batch_size = 1  # 批大小
num_layers = 1  # RNN层数


# 3. RNN模型
class WorkingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden


# 4. 初始化
model = WorkingRNN(input_size, hidden_size, input_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 训练循环
for epoch in range(100):
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    # 创建输入输出序列
    start_idx = np.random.randint(0, len(encoded) - seq_length)
    input_seq = encoded[start_idx:start_idx + seq_length]
    target_seq = encoded[start_idx + 1:start_idx + seq_length + 1]

    # 转换为one-hot (batch_size, seq_length, input_size)
    inputs = torch.zeros(batch_size, seq_length, input_size)
    for i in range(seq_length):
        inputs[0][i][input_seq[i]] = 1

    targets = torch.LongTensor(target_seq)

    # 训练步骤
    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs.squeeze(0), targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


# 6. 修正后的文本生成函数
def generate_text(start_char, length=10):
    with torch.no_grad():
        hidden = torch.zeros(num_layers, batch_size, hidden_size)

        # 初始化第一个字符输入 (batch_size=1, seq_length=1)
        current_input = torch.zeros(batch_size, 1, input_size)
        current_input[0][0][char_to_idx[start_char]] = 1

        output_text = start_char
        for _ in range(length):
            output, hidden = model(current_input, hidden)

            # 获取预测字符
            char_idx = torch.argmax(output[0, -1]).item()
            output_text += idx_to_char[char_idx]

            # 准备下一个输入 (保持batch_size和seq_length=1)
            current_input = torch.zeros(batch_size, 1, input_size)
            current_input[0][0][char_idx] = 1

        return output_text


# 7. 测试生成
print("\n生成文本:", generate_text('h'))