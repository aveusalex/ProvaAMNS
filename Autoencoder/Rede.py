import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        # Convolutional layers, 3 layers
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, 1000, stride=64, padding=0),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 100, stride=8, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 50, stride=4, padding=0),
            nn.ReLU(True),
            #nn.Conv1d(32, 64, 50, stride=8, padding=0),
            #nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(512, encoded_space_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.encoder_lin(x)
        #print(x.shape)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 16))

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose1d(64, 32, 50,
            #                    stride=8, padding=0, output_padding=0),
            # nn.BatchNorm1d(32),
            # nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, 50, stride=4,
                               padding=0, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 100, stride=8,
                               padding=0, output_padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 1856, stride=64,
                               padding=0, output_padding=0),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    from time import time
    # em gpu
    x = torch.randn(1, 1, 4*16000).cuda()
    print("Entrada: ", x.shape)
    encoder = Encoder(64).cuda()
    decoder = Decoder(64).cuda()
    x_ = encoder(x)
    x_ = decoder(x_)
    print("Saída:", x_.shape)
