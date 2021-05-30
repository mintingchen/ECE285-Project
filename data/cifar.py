import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 
import torch
import pickle
import matplotlib.pyplot as plt
import random

class CifarTrain(Dataset):

    def __init__(self, image_dir,image_list, mask_dir, mask_list, mask_reverse):
        super(CifarTrain, self).__init__()

        # cite https://www.cs.toronto.edu/~kriz/cifar.html
        def unpickle(file):
            with open(file, 'rb') as fo:
                myDict = pickle.load(fo, encoding='latin1')
            return myDict
        
        self.image_dir = image_dir
        self.images = unpickle(image_dir)['data']
        self.mask_dir = mask_dir
        self.masks = [x.strip() for x in open(mask_list)]
        self.mask_reverse = mask_reverse
     

        
#     def get_cifar_100(file_name):
#         with open(file_name, 'rb') as f:
#             batch_data = pickle.load(f, encoding='bytes')
#             batch_data[b"data"] = batch_data[b"data"] / 255
#         return batch_data[b"data"], batch_data[b"fine_labels"]

  
    def __getitem__(self, index):
        img = self.images[index]
        img = img.reshape(3,32,32).transpose(1,2,0)
        img = img.astype(np.float) / 255.0
        mask = cv2.imread(self.mask_dir+self.masks[int(random.random()*len(self.masks))], cv2.IMREAD_GRAYSCALE)
        
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask[mask>1] = 1
        if self.mask_reverse == "Yes":
            mask = np.logical_not(np.expand_dims(mask[:, :], axis=2))
        else:
            mask = np.expand_dims(mask[:, :], axis=2)
        masked_img = img * mask        

        img = torch.from_numpy(img).permute((2, 0, 1))
        mask = torch.from_numpy(mask).permute((2, 0, 1))
        masked_img = torch.from_numpy(masked_img).permute((2, 0, 1))

        return {
            'masked_img': masked_img.type(torch.float),
            'mask': mask.type(torch.float),
            'image': img.type(torch.float)
        }

    def __len__(self):
        return len(self.images)

class CifarTest(Dataset):

    def __init__(self, image_dir, image_list, mask_dir, mask_list, mask_reverse):
        super(CifarTest, self).__init__()

        def unpickle(file):
            with open(file, 'rb') as fo:
                myDict = pickle.load(fo, encoding='latin1')
            return myDict     
        self.image_dir = image_dir
        self.images = unpickle(image_dir)['data']
        self.mask_dir = mask_dir
        self.masks = [x.strip() for x in open(mask_list)]
        self.mask_reverse = mask_reverse

    def __getitem__(self, index):
        img = self.images[index]
        img = img.reshape(3,32,32).transpose(1,2,0)
        img = img.astype(np.float) / 255.0
        mask = cv2.imread(self.mask_dir+self.masks[index%len(self.masks)], cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask[mask>1] = 1
        if self.mask_reverse == "Yes":
            mask = np.logical_not(np.expand_dims(mask[:, :], axis=2))
        else:
            mask = np.expand_dims(mask[:, :], axis=2)
        masked_img = img * mask

        img = torch.from_numpy(img).permute((2, 0, 1))
        mask = torch.from_numpy(mask).permute((2, 0, 1))
        masked_img = torch.from_numpy(masked_img).permute((2, 0, 1))

        return {
            'masked_img': masked_img.type(torch.float),
            'mask': mask.type(torch.float),
            'image': img.type(torch.float)
        }

    def __len__(self):
        return len(self.images)