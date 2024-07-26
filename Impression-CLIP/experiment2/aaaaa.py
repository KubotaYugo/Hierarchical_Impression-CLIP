import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# データセットの準備
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# シンプルなニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# ロス関数とオプティマイザの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# TensorBoardのSummaryWriterを作成
writer = SummaryWriter('runs/')

# トレーニングループ
for epoch in range(5):  # 2エポックだけトレーニング
    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # パラメータの勾配をゼロにする
        optimizer.zero_grad()

        # 順伝播、逆伝播、オプティマイズ
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # ロスを記録
        running_loss += loss.item()
        
        # 精度の計算と記録
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        running_accuracy += accuracy

        if i % 100 == 99:  # 100ミニバッチごとにログを出力
            avg_loss = running_loss / 100
            avg_accuracy = running_accuracy / 100
            
            print(f'Epoch [{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}, accuracy: {avg_accuracy:.3f}')
            
            writer.add_scalar('training loss', avg_loss, epoch * len(trainloader) + i)
            writer.add_scalar('training accuracy', avg_accuracy, epoch * len(trainloader) + i)
            
            # 仮の値として学習率を記録
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning rate', current_lr, epoch * len(trainloader) + i)
            
            running_loss = 0.0
            running_accuracy = 0.0

print('Finished Training')

# TensorBoardログのクローズ
writer.close()