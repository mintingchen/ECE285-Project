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
import torch, argparse, pdb
from data import CreateDataloader
from model import CreateModel
from loss import VGG16PartialLoss
from metrics import compare_psnr, relMSE, PSNR, SSIM
from options import TestParser, set_init, print_options

def load_checkpoint(filename):
    ckpt = torch.load(filename, map_location='cpu')
    
    # model
    model = CreateModel(args)
    parameters = model.parameters() if gfilter is None else [{'params': model.parameters()}, {'params': gfilter.parameters()}]
    
    # optimizer = torch.optim.Adam(parameters, lr=0.0001, betas=(0.9, 0.99))
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
    rmse_inp_all, psnr_inp_all, ssim_inp_all = 0, 0, 0, 0, 0, 0
    
    with torch.no_grad():
        model.eval()
        for i, item in enumerate(dataset):
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            masked_img = data['masked_img'].to(device)
            image = data['image'].to(device)
            mask = data['mask'].to(device)  
            output, _ = model(image, mask)
            gt = image
                
                
#             inp = item['A'].cuda()
#             gt = item['B'].cuda()
            output = torch.squeeze(output.detach(), dim=0) * 255.0 # since batch = 1, squeeze the first channel
            output = torch.clamp(output, min = 0, max = 255)
            output = output.permute((1, 2, 0))
            output = output.cpu().numpy()
            gt = torch.squeeze(gt, dim=0) * 255.0 # since batch = 1, squeeze the first channel
            gt = torch.clamp(gt, min = 0, max = 255)
            gt = gt.permute((1, 2, 0))
            gt = gt.cpu().numpy()
            
            
            rmse = relMSE(output, gt)
            psnr = compare_psnr(output, gt, data_range=255.0)
            ssim = compare_ssim(output, gt, data_range=255.0, multichannel=True)
            if i % args.show_ratio == 0:
                cv2.imwrite('{}/{:05d}_output.png'.format(seq_path, i), output.astype(np.uint8))
                cv2.imwrite('{}/{:05d}_input.png'.format(seq_path, i), inp.astype(np.uint8))
                cv2.imwrite('{}/{:05d}_gt.png'.format(seq_path), i, gt.astype(np.uint8))
            rmse_inp_all += rmse
            psnr_inp_all += psnr
            ssim_inp_all += ssim
        rmse_inp_all /= i+1
        psnr_inp_all /= i+1
        ssim_inp_all /= i+1
        print("rmse: {:.4f}, psnr: {:.4f}, ssim: {:.4f}".format(rmse_inp_all, psnr_inp_all, ssim_inp_all))