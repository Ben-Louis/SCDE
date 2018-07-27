from .utils import *

class Discriminator(Module):
    def __init__(self, conv_dim=32, image_size=28, in_dim=1):
        super(Discriminator, self).__init__()
        
        layers = [ConvBlock(in_dim, conv_dim, gnorm=False, kernel_size=5, stride=1, padding=2)]
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



    
class Discriminator_asmb(model_asmb):
    def __init__(self, conv_dim=64, image_size=64):
        self.pho_net = Discriminator(conv_dim, image_size, 3)
        self.skt_net = Discriminator(conv_dim, image_size, 3)        
        
        self.map = {'pho':self.pho_net, 'skt':self.skt_net}
