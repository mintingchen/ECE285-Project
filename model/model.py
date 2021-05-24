import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from model.partconv2d import PartialConv2d



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_s=3, strides=2, encoding=False, bottleneck=False, decoding=False, relu=True, batchnorm=True):
        super(Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoding = encoding
        self.decoding = decoding
        self.bottleneck = bottleneck
        
        self.relu = relu
        self.batchnorm = batchnorm

        self.layers = None

        if self.encoding:
            self.l1 = PartialConv2d(in_channels, out_channels, kernel_s, strides, padding=kernel_s//2)
            if self.batchnorm:
                self.l2 = nn.Sequential(
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                self.l2 = nn.ReLU()   
                
        elif self.decoding:
            self.l1 = nn.Upsample(scale_factor=2, mode='nearest')
            self.l2 = PartialConv2d(in_channels, out_channels, kernel_s, strides, padding=kernel_s//2, mask_channels=2)
        
            if self.batchnorm:
                self.l3 = nn.Sequential(
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
   
                   
                
    def forward(self, img, mask, concat_img=0, concat_mask=0):
        if self.encoding:
            pconv_o, mask_o = self.l1(img, mask)
            pconv_o = self.l2(pconv_o)
            return pconv_o, mask_o

        if self.decoding:
            upimg_o = self.l1(img)
            upmask_o = self.l1(mask)
            concated_img = torch.cat((upimg_o, concat_img), 1)
            concated_mask = torch.cat((upmask_o, concat_mask), 1)
            pconv_o, mask_o = self.l2(concated_img, concated_mask)
            if self.batchnorm:
                pconv_o = self.l3(pconv_o)
            return pconv_o, mask_o
            

class Unet(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet, self).__init__()

        self.pconv1 = Block(in_channels, 64, kernel_s=7, encoding=True, batchnorm=False)
        self.pconv2 = Block(64, 128, kernel_s=5, encoding=True)
        self.pconv3 = Block(128, 256, kernel_s=5, encoding=True)
        self.pconv4 = Block(256, 512, encoding=True)
        self.pconv5 = Block(512, 512, encoding=True)
        self.pconv6 = Block(512, 512, encoding=True)
        self.pconv7 = Block(512, 512, encoding=True)
        self.pconv8 = Block(512, 512, encoding=True)
        
        self.pconv9 = Block(1024, 512, strides=1, decoding=True)
        self.pconv10 = Block(1024, 512, strides=1, decoding=True)
        self.pconv11 = Block(1024, 512, strides=1, decoding=True)
        self.pconv12 = Block(1024, 512, strides=1, decoding=True)
        self.pconv13 = Block(768, 256, strides=1, decoding=True)
        self.pconv14 = Block(384, 128, strides=1, decoding=True)
        self.pconv15 = Block(192, 64, strides=1, decoding=True)
        self.pconv16 = Block(67, 3, strides=1, decoding=True, batchnorm=False, relu=False)
        self.conv2d = nn.Conv2d(3, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, img, mask):
        pconv1_img, pconv1_mask = self.pconv1(img, mask)
        pconv2_img, pconv2_mask = self.pconv2(pconv1_img, pconv1_mask)
        pconv3_img, pconv3_mask = self.pconv3(pconv2_img, pconv2_mask)
        pconv4_img, pconv4_mask = self.pconv4(pconv3_img, pconv3_mask)
        pconv5_img, pconv5_mask = self.pconv5(pconv4_img, pconv4_mask)
        pconv6_img, pconv6_mask = self.pconv6(pconv5_img, pconv5_mask)
        pconv7_img, pconv7_mask = self.pconv7(pconv6_img, pconv6_mask)
        pconv8_img, pconv8_mask = self.pconv8(pconv7_img, pconv7_mask)
        
        pconv9_img, pconv9_mask = self.pconv9(pconv8_img, pconv8_mask, concat_img=pconv7_img, concat_mask=pconv7_mask)
        pconv10_img, pconv10_mask = self.pconv10(pconv9_img, pconv9_mask, concat_img=pconv6_img, concat_mask=pconv6_mask)
        pconv11_img, pconv11_mask = self.pconv11(pconv10_img, pconv10_mask, concat_img=pconv5_img, concat_mask=pconv5_mask)
        pconv12_img, pconv12_mask = self.pconv12(pconv11_img, pconv11_mask, concat_img=pconv4_img, concat_mask=pconv4_mask)
        pconv13_img, pconv13_mask = self.pconv13(pconv12_img, pconv12_mask, concat_img=pconv3_img, concat_mask=pconv3_mask)
        pconv14_img, pconv14_mask = self.pconv14(pconv13_img, pconv13_mask, concat_img=pconv2_img, concat_mask=pconv2_mask)
        pconv15_img, pconv15_mask = self.pconv15(pconv14_img, pconv14_mask, concat_img=pconv1_img, concat_mask=pconv1_mask)
        pconv16_img, pconv16_mask = self.pconv16(pconv15_img, pconv15_mask, concat_img=img, concat_mask=mask)
        output = self.relu(self.conv2d(pconv16_img))
        
        return output, pconv16_mask 
    
class Unet_light(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet_light, self).__init__()

        self.pconv1 = Block(in_channels, 64, kernel_s=7, encoding=True, batchnorm=False)
        self.pconv2 = Block(64, 128, kernel_s=5, encoding=True)
        self.pconv3 = Block(128, 256, kernel_s=5, encoding=True)
        self.pconv4 = Block(256, 512, encoding=True)
        self.pconv5 = Block(512, 512, encoding=True)
        
        self.pconv12 = Block(1024, 512, strides=1, decoding=True)
        self.pconv13 = Block(768, 256, strides=1, decoding=True)
        self.pconv14 = Block(384, 128, strides=1, decoding=True)
        self.pconv15 = Block(192, 64, strides=1, decoding=True)
        self.pconv16 = Block(67, 3, strides=1, decoding=True, batchnorm=False, relu=False)
        self.conv2d = nn.Conv2d(3, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, img, mask):
        pconv1_img, pconv1_mask = self.pconv1(img, mask)
        pconv2_img, pconv2_mask = self.pconv2(pconv1_img, pconv1_mask)
        pconv3_img, pconv3_mask = self.pconv3(pconv2_img, pconv2_mask)
        pconv4_img, pconv4_mask = self.pconv4(pconv3_img, pconv3_mask)
        pconv5_img, pconv5_mask = self.pconv5(pconv4_img, pconv4_mask)
        
        pconv12_img, pconv12_mask = self.pconv12(pconv5_img, pconv5_mask, concat_img=pconv4_img, concat_mask=pconv4_mask)
        pconv13_img, pconv13_mask = self.pconv13(pconv12_img, pconv12_mask, concat_img=pconv3_img, concat_mask=pconv3_mask)
        pconv14_img, pconv14_mask = self.pconv14(pconv13_img, pconv13_mask, concat_img=pconv2_img, concat_mask=pconv2_mask)
        pconv15_img, pconv15_mask = self.pconv15(pconv14_img, pconv14_mask, concat_img=pconv1_img, concat_mask=pconv1_mask)
        pconv16_img, pconv16_mask = self.pconv16(pconv15_img, pconv15_mask, concat_img=img, concat_mask=mask)
        output = self.relu(self.conv2d(pconv16_img))
        
        return output, pconv16_mask 