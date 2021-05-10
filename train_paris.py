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
import torch, argparse, pdb

from data import CreateDataloader

from tensorboardX import SummaryWriter
from options import TrainParser, set_seeds, print_options
import torchvision.utils as vutils
import cv2

def train(dataset, epoch, iter_num):
    for i, data in enumerate(dataset):
        iter_num += 1
        print(i)

        masked_img = data['masked_img']
        image = data['image']
        mask = data['mask']
        
        masked_img = masked_img[0].permute((1, 2, 0))
        masked_img = masked_img.cpu().numpy()
        image = image[0].permute((1, 2, 0))
        image = image.cpu().numpy()
        mask = mask[0].permute((1, 2, 0))
        mask = mask.cpu().numpy()
        cv2.imwrite("masked_img.png", masked_img)
        cv2.imwrite("image.png", image)
        cv2.imwrite("mask.png", mask)


if __name__ == '__main__':

    args_parser = TrainParser().parser
    args = args_parser.parse_args()
#     args = set_seeds(args)

    print_options(args)

    # dataset
    trainset, valset = CreateDataloader(args)

    iter_num = 0
    for epoch in range(args.epochs):
        print('\nEpoch %s' % (epoch+1))

        train(trainset, epoch, iter_num)

