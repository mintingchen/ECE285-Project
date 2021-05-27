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

class ParisTrain(Dataset):

    def __init__(self, image_dir, image_list, mask_dir, mask_list):
        super(ParisTrain, self).__init__()

        self.image_dir = image_dir
        self.images = [x.strip() for x in open(image_list)]
        self.mask_dir = mask_dir
        self.masks = [x.strip() for x in open(mask_list)]
        self.patch_height = 512
        self.patch_width = 512

    def __getitem__(self, index):
        img = cv2.imread(self.image_dir+self.images[index])
        mask = cv2.imread(self.mask_dir+self.masks[int(random.random()*len(self.masks))], cv2.IMREAD_GRAYSCALE)
        
        # The coordinate of the top-left corner for the croping patch 
        if img.shape[0] < self.patch_height or img.shape[1] < self.patch_width:
            if img.shape[0] < img.shape[1]:
                img = cv2.resize(img, (int(img.shape[1]/img.shape[0]*self.patch_width), self.patch_height))
            else:
                img = cv2.resize(img, (self.patch_width, int(img.shape[0]/img.shape[1]*self.patch_height)))
        
        img_crop_y = random.randint(0, img.shape[0] - self.patch_height)
        img_crop_x = random.randint(0, img.shape[1] - self.patch_width)

        img = img[img_crop_y : img_crop_y + self.patch_height, img_crop_x : img_crop_x + self.patch_width, :]
        img = img.astype(np.float) / 255.0
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

class ParisTest(Dataset):

    def __init__(self, image_dir, image_list, mask_dir, mask_list):
        super(ParisTest, self).__init__()

        self.image_dir = image_dir
        self.images = [x.strip() for x in open(image_list)]
        self.mask_dir = mask_dir
        self.masks = [x.strip() for x in open(mask_list)]
        self.patch_height = 512
        self.patch_width = 512

    def __getitem__(self, index):
        img = cv2.imread(self.image_dir+self.images[index])
        mask = cv2.imread(self.mask_dir+self.masks[index%len(self.masks)], cv2.IMREAD_GRAYSCALE)

        # The coordinate of the top-left corner for the croping patch 
        if img.shape[0] < self.patch_height or img.shape[1] < self.patch_width:
            if img.shape[0] < img.shape[1]:
                img = cv2.resize(img, (int(img.shape[1]/img.shape[0]*self.patch_width), self.patch_height))
            else:
                img = cv2.resize(img, (self.patch_width, int(img.shape[0]/img.shape[1]*self.patch_height)))
                
        img_crop_y = img.shape[0] // 2 - self.patch_height // 2
        img_crop_x = img.shape[1] // 2 - self.patch_width // 2

        img = img[img_crop_y : img_crop_y + self.patch_height, img_crop_x : img_crop_x + self.patch_width, :]
        img = img.astype(np.float) / 255.0
        mask = mask[mask_crop_y : mask_crop_y + self.patch_height, mask_crop_x : mask_crop_x + self.patch_width]
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