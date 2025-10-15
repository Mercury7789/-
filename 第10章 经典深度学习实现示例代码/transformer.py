import torch
import torch.nn as nn
import math
import numpy as np

# 1. 数据准备
text = "hello transformer"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)


# 2. 位置编码
# 2. 位置编码（修正版）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为缓冲区（增加batch维度）
        self.register_buffer('pe', pe.unsqueeze(0))  # 形状: [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 3. Transformer模型
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 16
        self.embed = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2),
            num_layers=1
        )
        self.fc = nn.Linear(self.d_model, vocab_size)

    def forward(self, src):
        src = self.embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        return self.fc(output)


# 4. 训练函数
def train_model():
    model = SimpleTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 准备数据
    encoded = [char_to_idx[c] for c in text]
    inputs = torch.LongTensor(encoded[:-1]).unsqueeze(1)  # [seq_len, 1]
    targets = torch.LongTensor(encoded[1:])  # [seq_len]

    # 训练循环
    for epoch in range(200):
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


# 5. 文本生成
def generate_text(model, start_str, length=10):
    model.eval()
    with torch.no_grad():
        # 初始化输入
        input_seq = [char_to_idx[c] for c in start_str]
        generated = start_str

        for _ in range(length):
            inputs = torch.LongTensor(input_seq).unsqueeze(1)  # [seq_len, 1]
            output = model(inputs)  # [seq_len, 1, vocab_size]

            # 获取最后一个预测字符
            last_char = torch.argmax(output[-1, 0]).item()
            generated += idx_to_char[last_char]
            input_seq.append(last_char)
            input_seq = input_seq[-len(start_str):]  # 保持上下文窗口

        return generated


# 6. 主程序
if __name__ == "__main__":
    # 训练模型
    print("开始训练...")
    trained_model = train_model()

    # 生成文本
    print("\n生成示例:")
    print(generate_text(trained_model, "hel", 10))
    print(generate_text(trained_model, "tra", 10))