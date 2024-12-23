import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(26, 64, kernel_size=3, stride=2, padding=1),  # Output: 64x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128x16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256x8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output: 512x4x4
            nn.ReLU()
        )
        self.fc = nn.Linear(512 * 4 * 4, 512)  # Flatten to 512-dim vector

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_type):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(512, 512 * 4 * 4)

        if decoder_type=='deconv':
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 256x8x8
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128x16x16
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 64x32x32
                nn.ReLU(),
                nn.ConvTranspose2d(64, 26, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 26x64x64
                nn.Sigmoid()  # Grayscale image range [0, 1]
            )
        elif decoder_type=='conv':
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Output: 256x8x8
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Output: 128x16x16
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),   # Output: 64x32x32
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 26, kernel_size=3, padding=1),    # Output: 26x64x64
                nn.Sigmoid()  # Grayscale image range [0, 1]
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to 4x4 feature map
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, decoder_type):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(decoder_type)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed