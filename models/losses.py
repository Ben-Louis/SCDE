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

        s = s.view(n, -1).unsqueeze(1).repeat(1, n, 1)
        pp = pp.view(n, -1).unsqueeze(0).repeat(n, 1, 1)

        dists = (s - pp).pow(2).sum(dim=2)
        dists = dists - dists.diag() + self.delta
        dists.clamp_(min=0.0)

        idx = torch.ones_like(dists)
        idx.sub_(idx.diag().diag())
        idx = idx != 0

        dists = dists.masked_select(idx).view(n, -1)

        max_loss = dists.max(dim=1)[0]
        mean_loss = dists.mean(dim=1)

        return max_loss, mean_loss