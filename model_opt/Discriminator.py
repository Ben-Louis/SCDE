import torch
import torch.nn as nn
from utils import *
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, conv_dim=32, image_size=128, in_dim=1):
        super(Discriminator, self).__init__()
        
        layers = [ConvBlock(in_dim, conv_dim, gnorm=False, kernel_size=7, stride=1, padding=3)]
        curr_dim = conv_dim
        curr_size = image_size
        while curr_size > 2:
            layers.append(ConvBlock(curr_dim, curr_dim*2, gnorm=False, kernel_size=4, stride=2, padding=1))
            curr_dim *= 2
            curr_size = curr_size // 2
        layers.append(ConvBlock(curr_dim, 1, False, False, kernel_size=3, stride=1, padding=1))
        self.main = Sequential(*layers)   
        
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
        return self.main(x)

    
class Discriminatorr(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=64, image_size=128, in_dim=3):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, 6):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, 6))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src#, out_cls.view(out_cls.size(0), out_cls.size(1))    
    
class Discriminator_asmb(model_asmb):
    def __init__(self, conv_dim=64, image_size=64):
        self.pho_net = Discriminator(conv_dim, image_size, 3)
        self.skt_net = Discriminator(conv_dim, image_size, 3)        
        
        self.map = {'pho':self.pho_net, 'skt':self.skt_net}
