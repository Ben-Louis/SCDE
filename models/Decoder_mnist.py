import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *




class Decoder(Module):
    def __init__(self, conv_dim, image_size, out_dim, large=False):
        super(Decoder, self).__init__()


        # in2
        layers = []
        curr_dim = conv_dim * 16
        img_size = 1

        self.fc = Sequential(
            nn.Linear(curr_dim, curr_dim*4),
            nn.ReLU(True))

        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(ConvBlock(curr_dim, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=False))        

        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(ConvBlock(curr_dim//2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=False))          
        layers.append(ConvBlock(conv_dim*2, conv_dim*2, kernel_size=2, stride=1, padding=0, bias=False))          
        curr_dim = conv_dim * 2
        self.deconv1 = Sequential(*layers)

        self.fushion1 = Sequential(
            AttentionLayer(curr_dim+conv_dim*8, curr_dim),
            ConvBlock(curr_dim+conv_dim*8, curr_dim, kernel_size=1, stride=1, padding=0, bias=False),
            ResidualBlock(curr_dim, curr_dim),
            ResidualBlock(curr_dim, curr_dim),
            nn.ReLU(True))

        self.deconv2 = Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(curr_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        curr_dim = conv_dim

        self.fushion2 = Sequential(
            AttentionLayer(curr_dim+conv_dim, curr_dim),
            ConvBlock(curr_dim+conv_dim, curr_dim, kernel_size=1, stride=1, padding=0, bias=False),
            ResidualBlock(curr_dim, curr_dim),
            ResidualBlock(curr_dim, curr_dim),
            nn.ReLU(True))

        self.deconv3 = Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            ConvBlock(curr_dim, 3, False, False, kernel_size=5, stride=1, padding=2, bias=False))
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()        

    def forward(self, z):
        z1 = z[0].unsqueeze(2).unsqueeze(3)
        z = torch.cat(z, dim=1)
        z = self.fc(z).view(z.size(0), -1, 2, 2)
        z = self.deconv1(z)

        z = torch.cat([z, z1.repeat(1,1,z.size(2),z.size(3))], dim=1)
        z = self.fushion1(z)
        z = self.deconv2(z)

        z = torch.cat([z, z1.repeat(1,1,z.size(2),z.size(3))], dim=1)
        z = self.fushion2(z)
        z = self.deconv3(z)        

        return z

class Decoder_asmb(model_asmb):
    def __init__(self, conv_dim=64, image_size=128, large=True):

        self.skt_net = Sequential(
            Decoder(conv_dim, image_size, 1, large),
            nn.Sigmoid())
        self.pho_net = Sequential(
            Decoder(conv_dim, image_size, 3, large),
            nn.Tanh())

        self.map = {'skt':self.skt_net, 'pho':self.pho_net}








