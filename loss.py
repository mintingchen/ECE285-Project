import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision import models

import os
os.environ['TORCH_HOME']='/home/xiwei/ece285/ECE285-Project'

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
            self.to_relu_1_2.add_module(str(4), nn.AvgPool2d(kernel_size=2, stride=2))
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x+1), features[x])
            self.to_relu_2_2.add_module(str(9), nn.AvgPool2d(kernel_size=2, stride=2))
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x+2), features[x])
            self.to_relu_3_3.add_module(str(16), nn.AvgPool2d(kernel_size=2, stride=2))
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x+3), features[x])
            self.to_relu_4_3.add_module(str(23), nn.AvgPool2d(kernel_size=2, stride=2))

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


# perceptual loss and (spatial) style loss
class VGG16PartialLoss():
    """
    VGG16 perceptual loss
    """
    def __init__(self, device, l1_alpha=5.0, perceptual_alpha=0.05, style_alpha=120,
                 smooth_alpha=0.1, feat_num=3, vgg_path='~/.torch/vgg16-397923af.pth'):
        """
        Init
        :param l1_alpha: weight of the l1 loss
        :param perceptual_alpha: weight of the perceptual loss
        :param style_alpha: weight of the style loss
        :param smooth_alpha: weight of the regularizer
        :param feat_num: number of feature maps
        """
        super().__init__()
        self.device = device

        self.vgg16partial = Vgg16().eval()
        self.vgg16partial.to(device)

        self.loss_fn = torch.nn.L1Loss(size_average=True)

        self.l1_weight = l1_alpha
        self.vgg_weight = perceptual_alpha
        self.style_weight = style_alpha
        self.regularize_weight = smooth_alpha

        self.dividor = 1
        self.feat_num = feat_num

    @staticmethod
    def normalize_batch(batch, div_factor=255.):
        """
        Normalize batch
        :param batch: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param div_factor: normalizing factor before data whitening
        :return: normalized data, tensor with shape
         (batch_size, nbr_channels, height, width)
        """
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch

    def __call__(self, output0, target0):
        """
        Forward
        assuming both output0 and target0 are in the range of [0, 1]
        :param output0: output of a model, tensor with shape
         (batch_size, nbr_channels, height, width)
        :param target0: target, tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: total perceptual loss
        """
        y = self.normalize_batch(target0, self.dividor)
        x = self.normalize_batch(output0, self.dividor)

        # L1 loss
        l1_loss = self.l1_weight * (torch.abs(x - y).mean())
        vgg_loss = 0
        style_loss = 0
        smooth_loss = 0

        # VGG
        if self.vgg_weight != 0 or self.style_weight != 0:

            yc = Variable(y.data)

            with torch.no_grad():
                groundtruth = self.vgg16partial(yc)
                generated = self.vgg16partial(x)
            
            # vgg loss: VGG content loss
            if self.vgg_weight > 0:
                # for m in range(0, len(generated)):
                for m in range(self.feat_num):

                    gt_data = Variable(groundtruth[m].data, requires_grad=False)
                    vgg_loss += (
                        self.vgg_weight * self.loss_fn(generated[m], gt_data)
                    )

            
            # style loss: Gram matrix loss
            if self.style_weight > 0:
                # for m in range(0, len(generated)):
                for m in range(self.feat_num):

                    gt_style = gram(
                        Variable(groundtruth[m].data, requires_grad=False))
                    gen_style = gram(generated[m])
                    style_loss += (
                        self.style_weight * self.loss_fn(gen_style, gt_style)
                    )

        # smooth term
        if self.regularize_weight != 0:
            smooth_loss += self.regularize_weight * (
                torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() +
                torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
            )

        tot = l1_loss + vgg_loss + style_loss + smooth_loss
        return tot, l1_loss, vgg_loss, style_loss, smooth_loss