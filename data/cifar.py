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

import random

class CifarTrain(Dataset):

    def __init__(self, image_dir, image_list, mask_dir, mask_list):
        super(CifarTrain, self).__init__()

        self.image_dir = image_dir
        self.images = [x.strip() for x in open(image_list)]
        self.mask_dir = mask_dir
        self.masks = [x.strip() for x in open(mask_list)]


    def __getitem__(self, index):
        img = cv2.imread(self.image_dir+self.images[index])
        mask = cv2.imread(self.mask_dir+self.masks[int(random.random()*len(self.masks))], cv2.IMREAD_GRAYSCALE)
        
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask[mask>1] = 1
        mask = np.logical_not(np.expand_dims(mask[:, :], axis=2))
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

    def __init__(self, image_dir, image_list, mask_dir, mask_list):
        super(CifarTest, self).__init__()

        self.image_dir = image_dir
        self.images = [x.strip() for x in open(image_list)]
        self.mask_dir = mask_dir
        self.masks = [x.strip() for x in open(mask_list)]

    def __getitem__(self, index):
        img = cv2.imread(self.image_dir+self.images[index])
        mask = cv2.imread(self.mask_dir+self.masks[index%len(self.masks)], cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask[mask>1] = 1
        mask = np.logical_not(np.expand_dims(mask[:, :], axis=2))
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