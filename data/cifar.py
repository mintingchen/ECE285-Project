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

    def __init__(self, image_dir, image_list, mask_dir):
        super(CifarTrain, self).__init__()

        self.image_dir = image_dir
        self.images = [x.strip() for x in open(image_list)]
        self.mask_dir = mask_dir
        self.masks = os.listdir(mask_dir)
        self.patch_height = 512
        self.patch_width = 512

    def __getitem__(self, index):
        img = cv2.imread(self.image_dir+self.images[index])
        mask = cv2.imread(self.mask_dir+self.masks[index], cv2.IMREAD_GRAYSCALE)
        
        # The coordinate of the top-left corner for the croping patch 
        img_crop_y = random.randint(0, img.shape[0] - self.patch_height)
        img_crop_x = random.randint(0, img.shape[1] - self.patch_width)
        mask_crop_y = random.randint(0, mask.shape[0] - self.patch_height)
        mask_crop_x = random.randint(0, mask.shape[1] - self.patch_width)

        img = img[img_crop_y : img_crop_y + self.patch_height, img_crop_x : img_crop_x + self.patch_width, :]
        mask = mask[mask_crop_y : mask_crop_y + self.patch_height, mask_crop_x : mask_crop_x + self.patch_width]
        mask[mask>1] = 1
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

    def __init__(self, image_dir, image_list, mask_dir, padding_factor=1):
        super(CifarTest, self).__init__()

        self.image_dir = image_dir
        self.images = [x.strip() for x in open(image_list)]
        self.mask_dir = mask_dir
        self.masks = os.listdir(mask_dir)
        self.padding_factor = padding_factor

    def __getitem__(self, index):
        img = cv2.imread(self.image_dir+self.images[index])
        mask = cv2.imread(self.mask_dir+self.masks[index], cv2.IMREAD_GRAYSCALE)

        img_height, img_width, _ = img.shape
        if img_width % self.padding_factor == 0:
            padded_img_width = img_width
        else:
            padded_img_width = (img_width // padding_factor + 1) * padding_factor

        if img_height % self.padding_factor == 0:
            padded_img_height = img_height
        else:
            padded_img_height = (img_height // padding_factor + 1) * padding_factor

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask[mask>1] = 1
        mask = np.expand_dims(mask[:, :], axis=2)
        masked_img = img * mask

        img_padded = np.zeros((padded_img_height, padded_img_width, 3), dtype=np.float)
        mask_padded = np.zeros((padded_img_height, padded_img_width, 3), dtype=np.float)
        masked_img_padded = np.zeros((padded_img_height, padded_img_width, 3), dtype=np.float)

        img_padded[:padded_img_height, :padded_img_width, :] = img
        mask_padded[:padded_img_height, :padded_img_width, :] = mask
        masked_img_padded[:padded_img_height, :padded_img_width, :] = masked_img


        img_padded = torch.from_numpy(img_padded).permute((2, 0, 1))
        mask_padded = torch.from_numpy(mask_padded).permute((2, 0, 1))
        masked_img_padded = torch.from_numpy(masked_img_padded).permute((2, 0, 1))

        return {
            'masked_img': masked_img_padded.type(torch.float),
            'mask': mask_padded.type(torch.float),
            'image': img_padded.type(torch.float)
        }

    def __len__(self):
        return len(self.images)