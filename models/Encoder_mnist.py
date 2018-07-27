import torch.nn.functional as F
from .utils import *


class Encoder(Module):

    def __init__(self, conv_dim=64, image_size=128, in_dim=3, large=False):
        super(Encoder, self).__init__()

        layers = []
        layers.append(ConvBlock(in_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=False))
        layers.append(ConvBlock(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        curr_dim = conv_dim*2
        for i in range(2):
            layers.append(ResidualBlock(curr_dim, curr_dim))
        layers.append(nn.LeakyReLU(0.01, True))
        self.main = Sequential(*layers)

        # out tube for z1
        self.out = Sequential(
            DenseBlock(curr_dim, curr_dim+curr_dim, downsample=True),
            ResidualBlock(curr_dim*2, curr_dim*2),
            ResidualBlock(curr_dim*2, curr_dim*2),
            nn.LeakyReLU(0.01, True),
            ConvBlock(curr_dim*2, curr_dim*4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvBlock(curr_dim*4, curr_dim*4, gnorm=False, lrelu=False, kernel_size=4, stride=1, padding=0, bias=False),)


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
        out = self.out(h).squeeze(3).squeeze(2)
        z1 = out[:,:32]
        z2 = out[:,32:]
        return z1, z2

class Encoder_asmb(model_asmb):
    def __init__(self, conv_dim=64, image_size=128, large=True):
        
        self.skt_net = Encoder(conv_dim, image_size, 3, large)
        self.pho_net = Encoder(conv_dim, image_size, 3, large)

        self.map = {'skt': self.skt_net, 'pho': self.pho_net}

    def __call__(self, z, mode, out=None):
        assert mode in self.map.keys()
        return self.map[mode](z)        

