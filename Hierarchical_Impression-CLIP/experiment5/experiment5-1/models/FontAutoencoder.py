import torch.nn as nn





class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                   nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class Encoder(nn.Module):
    def __init__(self, block, layers):
        super(Encoder, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Sequential(nn.Conv2d(26, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64),nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1 = self.make_layer(block, 32, layers[0], stride=1)   #down_sampleなし
        self.resblock2 = self.make_layer(block, 64, layers[1], stride=2)  #down_sampleあり
        self.enc_fc = nn.Sequential(nn.Linear(64*8*8, 512), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.view(-1, 64*8*8)
        x = self.enc_fc(x)
        return x

    def make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride), nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_fc = nn.Sequential(nn.Linear(512, 64*8*8), nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=26, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels=26, out_channels=26, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dec_fc(x)
        x = x.view(-1, 64, 8, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.sigmoid(x)
        return x



class Autoencoder(nn.Module):
    def __init__(self, block, layers):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(block, layers)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x