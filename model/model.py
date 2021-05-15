import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


class UNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 64
        super(UNet, self).__init__()
        self.PConv1 = self.encoder(3, 64, 7, batch_norm=False)
        self.PConv2 = self.encoder(64, 128, 5)
        self.PConv3 = self.encoder(128, 256, 5)
        self.PConv4 = self.encoder(256, 512, 3)
        self.PConv5 = self.encoder(512, 512, 3)
        self.PConv6 = self.encoder(512, 512, 3)
        self.PConv7 = self.encoder(512, 512, 3)
        self.PConv8 = self.encoder(512, 512, 3)
        
        self.UpSample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv9 = PartialConv2d(1024, 512, 3, strides=1, padding="same")
        
        self.UpSample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv10 = PartialConv2d(1024, 512, 3, strides=1, padding="same")
        
        self.UpSample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv11 = PartialConv2d(1024, 512, 3, strides=1, padding="same")
        
        self.UpSample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv12 = PartialConv2d(1024, 512, 3, strides=1, padding="same")
        
        self.UpSample5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv13 = PartialConv2d(768, 256, 3, strides=1, padding="same")
        
        self.UpSample6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv14 = PartialConv2d(384, 128, 3, strides=1, padding="same")
        
        self.UpSample7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv15 = PartialConv2d(192, 64, 3, strides=1, padding="same")
        
        self.UpSample8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.PConv16 = PartialConv2d(67, 3, 3, strides=1, padding="same")
        
    def encoder(self, in_channels, out_channels, kernel_s, batch_norm=True):
        layer = None
        if batch_norm == True:
            layer = nn.Sequential(
                    PartialConv2d(in_channels, out_channels, kernel_s, strides=2, padding="same")
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
        else:
            layer = nn.Sequential(
                    PartialConv2d(in_channels, out_channels, kernel_s, strides=2, padding="same")
                    nn.ReLU()
                )
        return layer
        
        
    def decoder(self, input_layer, concat_layer, in_channels, out_channels, kernel_s, batch_norm=True):
        layer = None
        if batch_norm == True:
            layer = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    concated_layer = torch.cat((input_layer, concat_layer), 2),
                    PartialConv2d(in_channels*2, out_channels, kernel_s, strides=1, padding="same"),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
        else:
            layer = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    concated_layer = torch.cat((input_layer, concat_layer), 2),
                    PartialConv2d(in_channels*2, out_channels, kernel_s, strides=1, padding="same"),
                    nn.LeakyReLU(0.2)
                )
        return layer
    
    def forward(self, x):
        h = x
        pc1 = self.PConv1(h)
        pc2 = self.PConv2(pc1)
        pc3 = self.PConv3(pc2)
        pc4 = self.PConv4(pc4)
        pc5 = self.PConv5(pc4)
        pc6 = self.PConv6(pc5)
        pc7 = self.PConv7(pc6)
        pc8 = self.PConv8(pc7)
        
        upsample1 = self.UpSample1(pc8)
        concat1 = torch.cat((upsample1, pc7), 2)
        pc9 = self.PConv9(concat1)
        pc9 = nn.BatchNorm2d(512)(pc9)
        pc9 = nn.LeakyReLU(0.2)(pc9)
        
        upsample2 = self.UpSample2(pc8)
        concat2 = torch.cat((upsample2, pc6), 2)
        pc10 = self.PConv10(concat2)
        pc10 = nn.BatchNorm2d(512)(pc10)
        pc10 = nn.LeakyReLU(0.2)(pc10)
        
        upsample3 = self.UpSample3(pc10)
        concat3 = torch.cat((upsample3, pc5), 2)
        pc11 = self.PConv11(concat3)
        pc11 = nn.BatchNorm2d(512)(pc11)
        pc11 = nn.LeakyReLU(0.2)(pc11)
        
        upsample4 = self.UpSample4(pc11)
        concat4 = torch.cat((upsample4, pc4), 2)
        pc12 = self.PConv12(concat4)
        pc12 = nn.BatchNorm2d(512)(pc12)
        pc12 = nn.LeakyReLU(0.2)(pc12)
        
        upsample5 = self.UpSample5(pc12)
        concat5 = torch.cat((upsample5, pc3), 2)
        pc13 = self.PConv13(concat5)
        pc13 = nn.BatchNorm2d(256)(pc13)
        pc13 = nn.LeakyReLU(0.2)(pc13)
        
        upsample6 = self.UpSample6(pc13)
        concat6 = torch.cat((upsample6, pc2), 2)
        pc14 = self.PConv14(concat6)
        pc14 = nn.BatchNorm2d(128)(pc14)
        pc14 = nn.LeakyReLU(0.2)(pc14)
        
        upsample7 = self.UpSample7(pc14)
        concat7 = torch.cat((upsample7, pc1), 2)
        pc15 = self.PConv15(concat7)
        pc15 = nn.BatchNorm2d(64)(pc15)
        pc15 = nn.LeakyReLU(0.2)(pc15)
        
        upsample8 = self.UpSample8(pc15)
        concat8 = torch.cat((upsample8, x), 2)
        pc16 = self.PConv16(concat8)
        
        return pc16

        
#     def encoder(x, x_mask, channels, kernel_s, batch_norm=True):
#         x, x_mask = PartialConv2d(channels, kernel_s, strides=2, padding="same")([x, x_mask])
#         if batch_norm == True:
#             x = nn.BatchNorm2d(channels)(x)
#         x = nn.ReLU()(x)
#         return x, x_mask
#     def decoder(x, x_mask, z, z_mask, channels, kernel_s, batch_norm=True):
#         x = nn.Upsample(scale_factor=2, mode='nearest')(x)
#         x_mask = nn.Upsample(scale_factor=2, mode='nearest')(x_mask)
#         x_concat = torch.cat((x, z), 2) 
#         mask_concat = torch.cat((x_mask, z_mask), 2) 
#         x, x_mask = PartialConv2d(channels, kernel_s, strides=1, padding="same")([x_concat, mask_concat])
#         if batch_norm == True:
#             x = nn.BatchNorm2d(channels)(x)
#         x = nn.LeakyReLU(0.2)(x)
#         return x, x_mask

       
        
        

