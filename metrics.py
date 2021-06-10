import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from sklearn.metrics import mean_squared_error

def l1_loss(img1, img2, data_range=255.0):
#     img1 = img1 / data_range
#     img2 = img2 / data_range
    err = abs(img1.astype("float") - img2.astype("float")).mean()
    
    return err

def compare_psnr(img1, img2, data_range=255.0):
    mse = np.mean((img1/data_range - img2/data_range)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def relMSE(pred, gt, eps=0.01, data_range=255.0):
    pred = pred / data_range
    gt = gt / data_range
    err = (pred.astype("float") - gt.astype("float"))**2
    rmse = (err / (gt.astype("float")**2 + eps)).mean()
    
    return rmse