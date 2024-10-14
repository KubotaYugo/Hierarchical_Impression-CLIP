import torch.nn as nn
from torchvision import models


class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(26, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.resnet(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=26, kernel_size=4, stride=2, padding=1))
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(in_channels=26, out_channels=26, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dec_fc(x)
        x = x.view(-1, 512, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.sigmoid(x)
        return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = ModifiedResNet()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# from torchinfo import summary

# model = autoencoder()
# summary(model, (10, 26, 64, 64))