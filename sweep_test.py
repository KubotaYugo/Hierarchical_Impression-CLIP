import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# データセットとデータローダーの準備
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# シンプルなモデル定義
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの訓練と評価関数
def train_and_evaluate(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # データローダーの設定
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # モデル、損失関数、オプティマイザの設定
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()

        if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

        # 訓練ループ
        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 検証ループ
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            wandb.log({"validation_accuracy": val_acc})
            print(f'Epoch [{epoch+1}/5], Validation Accuracy: {val_acc:.4f}')

# Sweep設定
sweep_config = {
    'method': 'random',  # ランダム探索
    'metric': {'name': 'validation_accuracy', 'goal': 'maximize'},  # 精度を最大化
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'batch_size': {'values': [16, 32, 64]},
        'optimizer': {'values': ['adam', 'sgd']}
    }
}

# Sweepの作成
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweep-sample")

# Sweepの実行
wandb.agent(sweep_id, train_and_evaluate, count=10)  # 10回の実験を実行