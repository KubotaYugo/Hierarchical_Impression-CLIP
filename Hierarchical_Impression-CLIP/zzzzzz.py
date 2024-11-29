import torch
import torch.nn as nn

# サンプル数
num_samples = 100000

# 一様乱数 [-1, 1] のロジットを生成
logits = torch.empty(num_samples).uniform_(-1, 1)

# ランダムバイナリラベル
labels = torch.randint(0, 2, (num_samples,))

# BCEWithLogitsLoss を定義
criterion = nn.BCEWithLogitsLoss()

# 損失を計算
loss = criterion(logits, labels.float())

print(f"Average BCE Loss: {loss.item():.4f}")
