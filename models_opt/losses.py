import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, delta=0.3, dist=None):
        super(TripletLoss, self).__init__()

        self.delta = delta
        if dist == None:
            dist = lambda z1, z2: torch.norm(z1-z2, p=2, dim=1)
        self.dist = dist

    def forward(self, s, pp, pn):
        s  = F.normalize(s )
        pp = F.normalize(pp)
        pn = F.normalize(pn)


        dp = self.dist(s, pp)
        dn = self.dist(s, pn)
        dist = self.delta+dp-dn
        dist = torch.clamp(dist, min=0.0)

        return torch.mean(dist) / 2.0