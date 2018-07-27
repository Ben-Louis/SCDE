import os
import torch
import torchvision as tv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import json
import numpy as np

def merge_list(l):
    res = []
    for elem in l:
        res.extend(elem)
    return res


class MnistSvhnData(Dataset):
    def __init__(self, data_root, random=True):
        
        mnist_root = os.path.join(data_root, 'mnist')
        svhn_root = os.path.join(data_root, 'svhn')

        self.mnist = tv.datasets.MNIST(mnist_root, download=True)
        self.svhn = tv.datasets.SVHN(svhn_root, download=True)

        with open(os.path.join(data_root, 'mnistsvhn.json'), 'r') as f:
            self.coord = json.load(f)
        self.coord = merge_list(self.coord)

        self.to_tensor = transforms.ToTensor()
        self.random = random

        
    def __len__(self):
        return len(self.coord)
    
    def __getitem__(self, index):
        idxm, idxs = self.coord[index]
        imgm = self.to_tensor(self.mnist[idxm][0])
        w = imgm.size(1)
        if self.random:
            color = np.random.randint(2, size=(3,1))
            while np.sum(color) == 0:
                color = np.random.randint(2, size=(3,1))
            color = torch.FloatTensor(color)
        else:
            color = torch.ones(3,1)
        imgm = color.matmul(imgm.view(1,-1)).view(3, w, w)
        imgs = self.to_tensor(self.svhn[idxs][0].resize((28,28))) * 2 - 1
        
        return imgm, imgs

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)
