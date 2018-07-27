import torch.nn.functional as F
from .utils import *


class Encoder(Module):

    def __init__(self, conv_dim=64, image_size=128, in_dim=3, large=False):
        super(Encoder, self).__init__()

        layers = []
        layers.append(ConvBlock(in_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(ConvBlock(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        curr_dim = conv_dim*2
        for i in range(3):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        layers.append(nn.LeakyReLU(0.01, True))
        self.main = Sequential(*layers)

        # out tube for z1
        self.out1 = Sequential(
            ConvBlock(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(curr_dim*2, curr_dim*2),
            ResidualBlock(curr_dim*2, curr_dim*2),
            ResidualBlock(curr_dim*2, curr_dim*2),
            ResidualBlock(curr_dim*2, curr_dim*2),
            nn.LeakyReLU(0.01, True),
            ConvBlock(curr_dim*2, curr_dim*2, gnorm=False, lrelu=False, kernel_size=3, stride=1, padding=1, bias=False))

        # out tube for z2
        if not large:
            self.z2_conv = ConvBlock(curr_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
            curr_dim = conv_dim*2 + 3
            image_size = image_size // 4
            layers = []
            for i in range(3):
                layers.append(DenseBlock(curr_dim, curr_dim+conv_dim, downsample=True))
                curr_dim += conv_dim
                image_size = image_size // 2
            layers.append(ConvBlock(curr_dim, conv_dim*3, False, False, kernel_size=1, stride=1, padding=0, bias=False))

        else:
            self.z2_conv = ConvBlock(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
            curr_dim = conv_dim*4 + 3
            image_size = image_size // 4
            layers = []
            for i in range(3):
                layers.append(DenseBlock(curr_dim, curr_dim+conv_dim*2, downsample=True))
                curr_dim += conv_dim*2
                image_size = image_size // 2            
            layers.append(ConvBlock(curr_dim, curr_dim, False, False, kernel_size=1, stride=1, padding=0, bias=False))
        self.out2 = Sequential(*layers)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()        
        
    def forward(self, x):
        h = self.main(x)
        z1 = self.out1(h)
        z2 = self.out2(torch.cat([self.z2_conv(h), F.max_pool2d(x, 4,4)], dim=1))
        return z1, z2

class Encoder_asmb(model_asmb):
    def __init__(self, conv_dim=64, image_size=128, large=True):
        
        self.skt_net = Encoder(conv_dim, image_size, 3, large)
        self.pho_net = Encoder(conv_dim, image_size, 3, large)

        self.map = {'skt': self.skt_net, 'pho': self.pho_net}

    def __call__(self, z, mode, out=None):
        assert mode in self.map.keys()
        return self.map[mode](z)        

