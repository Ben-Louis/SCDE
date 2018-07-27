import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, delta=0.3, dist=None):
        super(TripletLoss, self).__init__()

        #self.abalation_tri = abalation_tri.split(',')

        self.delta = delta
        if dist == None:
            dist = lambda z1, z2: (z1-z2).pow(2).sum(dim=1)
        self.dist = dist

    def forward(self, s, pp, pn):
        #s  = F.normalize(s )
        #pp = F.normalize(pp)
        #pn = F.normalize(pn)
        n = s.size(0)

        s = F.normalize(s.view(n, -1)).unsqueeze(1).repeat(1, n, 1)
        pp = F.normalize(pp.view(n, -1)).unsqueeze(0).repeat(n, 1, 1)

        dists = (s - pp).pow(2).sum(dim=2)
        dists = -(dists - dists.diag()) + self.delta
        dists.clamp_(min=0.0)

        idx = torch.ones_like(dists)
        idx.sub_(idx.diag().diag())
        idx = idx != 0

        dists = dists.masked_select(idx).view(n, -1)

        max_loss = dists.max(dim=1)[0].mean()
        mean_loss = dists.mean()

        return max_loss, mean_loss

class SDLLoss:
    def __init__(self, dim1, dim2, alpha=0.5, S=None):
        super(SDLLoss, self).__init__()
        if S is not None:
            self.S = S
        else:
            S = torch.ones((dim1+dim2, dim1+dim2))
            S[:dim1, :dim1] = 0
            S[dim1:, dim1:] = 0
            self.S = S
        self.S.requires_grad_(False)
        self.c = 0
        self.alpha = alpha
        self.Caccu = torch.zeros((dim1+dim2, dim1+dim2))
        self.Caccu.requires_grad_(False)
        #self.low_bound = torch.FloatTensor([50])

    def to(self, device):
        print('SDL at device!')
        self.S = self.S.to(device)
        self.Caccu = self.Caccu.to(device)
        #self.low_bound = self.low_bound.to(device)

    def __call__(self, z1, z2):
        z = torch.cat([z1, z2], dim=1)
        # shape of z: (B, k)
        mean = torch.mean(z, dim=0)
        var = torch.var(z, dim=0) + 1e-8
        z = (z-mean)/var

        Cmini = z.t().mm(z) / z.size(0)
        #print(Cmini.shape, self.Caccu.shape)
        self.Caccu = self.alpha * self.Caccu.detach() + Cmini
        self.c = self.alpha * self.c + 1
        Cappx = self.Caccu / self.c
        loss = torch.sum(torch.abs(Cappx*self.S))
        return loss  
