import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision import models

import os
os.environ['TORCH_HOME']='/home/xiwei/ece285/ECE285-Project'

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
        self.pool1 = nn.Sequential()
        self.pool2 = nn.Sequential()
        self.pool3 = nn.Sequential()
        self.pool4 = nn.Sequential()

        for x in range(5):
            self.pool1.add_module(str(x), features[x])
        for x in range(5, 10):
            self.pool2.add_module(str(x), features[x])
        for x in range(10, 17):
            self.pool3.add_module(str(x), features[x])
        for x in range(17, 24):
            self.pool4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.pool1(x)
        h_pool1 = h
        h = self.pool2(h)
        h_pool2 = h
        h = self.pool3(h)
        h_pool3 = h
        h = self.pool4(h)
        h_pool4 = h
        out = (h_pool1, h_pool2, h_pool3, h_pool4)
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
    def __init__(self, device, l1_alpha=6.0, perceptual_alpha=0.05, style_alpha=240,
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

    def __call__(self, output0, target0, mask):
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
        l1_loss = self.l1_weight * (torch.abs((1-mask)*x - (1-mask)*y).mean()) + (torch.abs(mask*x - mask*y).mean())
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
