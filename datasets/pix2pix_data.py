import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class Pix2PixData(Dataset):
    def __init__(self, data_root, obj='shoes', mode='train', image_size=128, random=False, part=False):
        
        self.image_size = image_size
        self.threshold = 0.75
        if image_size == 64:
            self.threshold = 0.87
        
        # dir
        assert obj in ['shoes', 'handbags']
        #root = os.path.join(os.environ['HOME'], 'dataset', 'pix2pix', 'datasets', 'edges2'+obj)
        root = data_root
        train_dir = os.path.join(root, 'train')
        valid_dir = os.path.join(root, 'val')
    
        # files
        train_files = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir)]
        valid_files = [os.path.join(valid_dir, fname) for fname in os.listdir(valid_dir)]
        if part:
            train_files = train_files[:300]
        if mode == 'train':
            self.files = train_files
        else:
            self.files = valid_files                
        
        # transforms
        self.toTensor = transforms.Compose([
            transforms.Resize((image_size, image_size*2),Image.BILINEAR),
            transforms.ToTensor()])
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.random = random
        if random:
            self.kernel = np.ones((2,2),np.uint8)
            self.iter = 3 
            if image_size == 64:
                self.iter = 2

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = Image.open(self.files[index])

        img = self.toTensor(img)
        
        skt, pho = img[:,:,:self.image_size], img[:,:,self.image_size:]


        skt = 1-(skt>self.threshold).type(torch.FloatTensor)
        if self.random:
            width = np.random.randint(self.iter)
            skt = cv2.dilate(skt[0].numpy(),self.kernel,iterations = width)
            skt = torch.from_numpy(skt)
            
            w = skt.size(1)
            color = np.random.randint(2, size=(3,1))
            while np.sum(color) == 0:
                color = np.random.randint(2, size=(3,1))
            color = torch.FloatTensor(color)
            skt = color.matmul(skt.view(1,-1)).view(3, w, w)
                
            
        pho = self.norm(pho)
        
        return skt, pho

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)
