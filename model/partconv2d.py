import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, dilation=1, mask_channels=1):
        super(PartialConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_channels = mask_channels
        self.kernel_size = [kernel_size, kernel_size]
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding, bias=self.bias)

#         if self.mask_channels > 1:
#             self.weight_maskUpdater = torch.ones(self.out_channels, self.mask_channels, self.kernel_size[0], self.kernel_size[1])
#         else:
        self.weight_maskUpdater = torch.ones(1, self.mask_channels, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in):
        assert len(input.shape) == 4
        self.last_size = tuple(input.shape)

        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)

            mask = mask_in
                    
            self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

            # for mixed precision training, change 1e-8 to 1e-6
            self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
            # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        if self.mask_channels > 1:
            masked_img1 = torch.mul(input[:,:input.shape[1]//2,...], mask[:,0,...].unsqueeze(1))
            masked_img2 = torch.mul(input[:,input.shape[1]//2:,...], mask[:,1,...].unsqueeze(1))
            raw_out = self.conv(torch.cat((masked_img1, masked_img2), 1))
        else:
            raw_out = self.conv(torch.mul(input, mask))

#         if self.bias is not None:
#             bias_view = self.bias.view(1, self.out_channels, 1, 1)
#             output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
#             output = torch.mul(output, self.update_mask)
#         else:
        output = torch.mul(raw_out, self.mask_ratio)


#         if self.return_mask:
#             return output, self.update_mask
#         else:
        return output, self.update_mask
        