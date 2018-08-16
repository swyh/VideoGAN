import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64 * 8, 100, 4, 1, 0, bias=False),
        nn.BatchNorm2d(100),
        nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(64 * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        nn.Sigmoid()

        self.main = nn.Sequential(
        #inner / outer / kernal, stride, padding
            nn.Conv2d(12, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 100, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(100, 512, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main( input )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(256, 512, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(512, 1, 4, 1, 0),
        nn.Sigmoid()
        )

    def forward(self, input):
        return self.main( input )


