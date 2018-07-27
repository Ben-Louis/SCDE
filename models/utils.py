import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Module(nn.Module):
    def load_state_dict(self, state_dict, strict=False):
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    pass
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
                
class Sequential(nn.Sequential):
    def load_state_dict(self, state_dict, strict=False):
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    pass
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))                

class ConvBlock(Module):
    def __init__(self, dim_in, dim_out, gnorm=True, lrelu=True, **kwargs):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(dim_in, dim_out, **kwargs))
        if gnorm:
            layers.append(nn.GroupNorm(32, dim_out))
        if lrelu:
            layers.append(nn.LeakyReLU(0.01, inplace=True))

        self.main = Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class DenseBlock(Module):
    def __init__(self, dim_in, dim_out, downsample=False, out=False):
        super(DenseBlock, self).__init__()
        
        gnorm, lrelu = (not out, not out)

        if downsample:
            self.conv = ConvBlock(dim_in, dim_out-dim_in, gnorm=gnorm, lrelu=lrelu, kernel_size=4, stride=2, padding=1, bias=False)
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.conv = ConvBlock(dim_in, dim_out-dim_in, gnorm=gnorm, lrelu=lrelu, kernel_size=3, stride=1, padding=1, bias=False)
            self.downsample = lambda x: x
        
    def forward(self, x):
        h = self.conv(x)
        x = self.downsample(x)
        return torch.cat([x,h], dim=1)
        
class ResidualBlock_cus(Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ResidualBlock, self).__init__()
        self.main = Sequential(
            nn.Conv2d(dim_in, dim_out, **kwargs),
            nn.GroupNorm(32, dim_out),
            nn.LeakyReLU(0.01,inplace=True),
            nn.Conv2d(dim_out, dim_out, **kwargs),
            nn.GroupNorm(32, dim_out))

    def forward(self, x):
        return x + self.main(x)    
    
class ResidualBlock(Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ResidualBlock, self).__init__()
        self.main = Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False, **kwargs),
            nn.GroupNorm(32, dim_out),
            nn.LeakyReLU(0.01,inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False, **kwargs),
            nn.GroupNorm(32, dim_out))

    def forward(self, x):
        return x + self.main(x)
    
class model_asmb:
    def __init__(self):
        self.skt_net = None
        self.pho_net = None
        self.map = {'pho':self.pho_net, 'skt':self.skt_net}
    
    def __call__(self, z, mode):
        assert mode in self.map.keys()
        return self.map[mode](z)

    def parameters(self):
        return [*self.pho_net.parameters(), *self.skt_net.parameters()]

    def save_model(self, path):
        path, tp = path.split('.')
        path_skt = path + '_skt.' + tp
        path_pho = path + '_pho.' + tp
        torch.save(self.skt_net.state_dict(), path_skt)
        torch.save(self.pho_net.state_dict(), path_pho)

    def load_model(self, path):
        path, tp = path.split('.')
        path_skt = path + '_skt.' + tp
        path_pho = path + '_pho.' + tp
        self.skt_net.load_state_dict(torch.load(path_skt), strict=False)
        self.pho_net.load_state_dict(torch.load(path_pho), strict=False)

    def to(self, device):
        self.pho_net = self.pho_net.to(device)
        self.skt_net = self.skt_net.to(device)


class AttentionLayer(Module):
    def __init__(self, conv_dim, mod_dim=0):
        super(AttentionLayer, self).__init__()

        if mod_dim > 0:
            self.mod_dim = mod_dim
        else:
            self.mod_dim = conv_dim

        self.conv1 = nn.Conv2d(conv_dim, conv_dim//8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim//8, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(conv_dim, mod_dim, kernel_size=1, stride=1, padding=0)

        self.alpha = Parameter(torch.FloatTensor([0.0]))
        self.beta = Parameter(torch.FloatTensor([1.0]))

    def to(self, device):
        self.conv1.to(device)
        self.conv2.to(device)
        self.conv3.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha.requires_grad_(True)
        self.beta = self.beta.to(device)
        self.beta.requires_grad_(True)

    def forward(self, x):
        chn = x.size(1)
        f, g = self.conv1(x), self.conv2(x)
        f = f.view(f.size(0), f.size(1), -1)
        g = g.view(g.size(0), g.size(1), -1)

        w = torch.matmul(torch.transpose(f,1,2),g)

        # normalize
        w = F.softmax(w, dim=1)

        # activation
        h = self.conv3(x)
        h = h.view(x.size(0), self.mod_dim, -1)
        x_mod, x_reserve = x[:,:self.mod_dim,:,:], x[:,self.mod_dim:,:,:]
        h = torch.matmul(h, w).view(x_mod.shape)

        out_mod = self.alpha * h + self.beta * x_mod

        return torch.cat([out_mod, x_reserve], dim=1)        
