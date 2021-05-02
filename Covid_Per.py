from torchvision.datasets import MNIST

import PIL 
import os
import os.path
import numpy as np
import torch
import cv2
import torch.nn.functional as F

class Covid_Per(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.targets, self.sub = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, sub = self.data[index], self.targets[index], self.sub[index]
        img = np.array(img)
        img = img.astype(np.uint8)
                       
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return  img, target, sub

    def __len__(self):
        return len(self.data)
    
