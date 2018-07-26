import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils import *




class Decoder(Module):
    def __init__(self, conv_dim, image_size, out_dim, large=True):
        super(Decoder, self).__init__()


        # in2
        layers = []
        img_size = image_size // 64

        if not large:
            curr_dim = (conv_dim*3)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.deconv1 = Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvBlock(curr_dim+3, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))

            curr_dim = curr_dim + conv_dim * 4
            self.deconv2 = ConvBlock(curr_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=False)

            curr_dim = conv_dim * 6
            self.deconv3 = ConvBlock(curr_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=False)
                    
            # main
            curr_dim = conv_dim * 4
            layers = [ConvBlock(conv_dim*6, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)]
        else:
            curr_dim = (conv_dim*14)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            #self.deconv1 = Sequential(
            #    nn.Upsample(scale_factor=2, mode='nearest'),
            #    ConvBlock(curr_dim+3, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=False))

            #curr_dim = curr_dim//2 + conv_dim * 4
            self.deconv2 = ConvBlock(curr_dim+3, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=False)

            curr_dim =  conv_dim * 10
            self.deconv3 = ConvBlock(curr_dim, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=False)
                    
            # main
            curr_dim = conv_dim * 4
            layers = [ConvBlock(curr_dim*2, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)]            



        for i in range(4):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(ConvBlock(curr_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=False))            
            
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(ConvBlock(conv_dim*2, conv_dim, kernel_size=5, stride=1, padding=2, bias=False))
        if out_dim == 3:
            layers.append(ConvBlock(conv_dim, 3, False, False, kernel_size=7, stride=1, padding=3, bias=False))
        else:
            layers.append(ConvBlock(conv_dim, 3, False, False, kernel_size=3, stride=1, padding=1, bias=False))

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

    def forward(self, z):
        z1, z2 = z
        #z2 = self.deconv1(z2)
        z2 = torch.cat([F.avg_pool2d(z1, 4, 4), self.up(z2)], dim=1)
        z2 = self.deconv2(z2)
        z2 = torch.cat([F.avg_pool2d(z1, 2, 2), self.up(z2)], dim=1)
        z2 = self.deconv3(z2)
        z = torch.cat([z1, self.up(z2)], dim=1)
        return self.main(z)

class Decoder_asmb(model_asmb):
    def __init__(self, conv_dim=64, image_size=128, large=True):

        self.skt_net = Sequential(
            Decoder(conv_dim, image_size, 1, large),
            nn.Sigmoid())
        self.pho_net = Sequential(
            Decoder(conv_dim, image_size, 3, large),
            nn.Tanh())

        self.map = {'skt':self.skt_net, 'pho':self.pho_net}








