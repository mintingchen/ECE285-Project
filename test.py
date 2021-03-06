from skimage.metrics import structural_similarity as compare_ssim
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
from numpy import *
import torch, argparse, pdb
from data import CreateDataloader
from model import CreateModel
from loss import VGG16PartialLoss
from metrics import compare_psnr, l1_loss
from options import TestParser, set_init, print_options
from tqdm import tqdm

def cal_mask_ratio(mask):
    valid = torch.sum(mask).item()
    total = mask.numel()
    mask_ratio = (total-valid)/total
    if mask_ratio < 0.1:
        return 0
    elif mask_ratio < 0.2:
        return 1
    elif mask_ratio < 0.3:
        return 2
    elif mask_ratio < 0.4:
        return 3
    elif mask_ratio < 0.5:
        return 4
    else:
        return 5

def load_checkpoint(filename):
    ckpt = torch.load(filename, map_location='cpu')
    
    # model
    model = CreateModel(args)
    parameters = model.parameters() 
    #if gfilter is None else [{'params': model.parameters()}, {'params': gfilter.parameters()}]
    
    # optimizer = torch.optim.Adam(parameters, lr=0.0001, betas=(0.9, 0.99))
    #print(ckpt['state_dict'])
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict']) # load 
    # optimizer.load_state_dict(ckpt['optimizer'])
    model.cuda()
    
    return model, int(epoch)

if __name__ == '__main__':
    args_parser = TestParser().parser
    args = args_parser.parse_args()
    
    # dataset
    dataset = CreateDataloader(args, mode='test')
    print(args.checkpoint)
    model, epoch = load_checkpoint(args.checkpoint)
    seq_path = args.seq_path
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    l1_inp_all, psnr_inp_all, ssim_inp_all = 0, 0, 0
    l1_dict, psnr_dict, ssim_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}, {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}, {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    with torch.no_grad():
        model.eval()
        for i, data in tqdm(enumerate(dataset)):
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            masked_img = data['masked_img'].to(device)
            image = data['image'].to(device)
            mask = data['mask'].to(device)  
            output, _ = model(image, mask)
            gt = image
            
            mask_type = cal_mask_ratio(mask)
                
            masked_img = torch.squeeze(masked_img.detach(), dim=0) * 255.0 # since batch = 1, squeeze the first channel
            masked_img = torch.clamp(masked_img, min = 0, max = 255)
            masked_img = masked_img.permute((1, 2, 0))
            masked_img = masked_img.cpu().numpy()
    
            output = torch.squeeze(output.detach(), dim=0) * 255.0 # since batch = 1, squeeze the first channel
            output = torch.clamp(output, min = 0, max = 255)
            output = output.permute((1, 2, 0))
            output = output.cpu().numpy()
            gt = torch.squeeze(gt, dim=0) * 255.0 # since batch = 1, squeeze the first channel
            gt = torch.clamp(gt, min = 0, max = 255)
            gt = gt.permute((1, 2, 0))
            gt = gt.cpu().numpy()
            
            
            l1 = l1_loss(output, gt)
            psnr = compare_psnr(output, gt, data_range=255.0)
            ssim = compare_ssim(output, gt, data_range=255.0, multichannel=True)
            if i % args.show_ratio == 0:
                cv2.imwrite('{}/{:05d}_mask_{:02d}_output.png'.format(seq_path, i, mask_type), output.astype(np.uint8))
                cv2.imwrite('{}/{:05d}_mask_{:02d}_input.png'.format(seq_path, i, mask_type), masked_img.astype(np.uint8))
                cv2.imwrite('{}/{:05d}_mask_{:02d}_gt.png'.format(seq_path, i, mask_type), gt.astype(np.uint8))
            l1_inp_all += l1
            psnr_inp_all += psnr
            ssim_inp_all += ssim
            
            l1_dict[mask_type].append(l1)
            psnr_dict[mask_type].append(psnr)
            ssim_dict[mask_type].append(ssim)
        l1_inp_all /= i+1
        psnr_inp_all /= i+1
        ssim_inp_all /= i+1
        for i in range(6):
            print("ratio type: {} => Num: {}".format(i, len(l1_dict[i])))
            l1_dict[i] = mean(l1_dict[i])
            psnr_dict[i] = mean(psnr_dict[i])
            ssim_dict[i] = mean(ssim_dict[i])
            print("\t l1:", l1_dict[i])
            print("\t psnr:", psnr_dict[i])
            print("\t ssim:", ssim_dict[i])
        
        print("l1_loss: {:.4f}, psnr: {:.4f}, ssim: {:.4f}".format(l1_inp_all, psnr_inp_all, ssim_inp_all))