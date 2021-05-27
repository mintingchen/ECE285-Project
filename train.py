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

from tensorboardX import SummaryWriter
from options import TrainParser, set_init, print_options
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau


from data import CreateDataloader
from model import CreateModel
from loss import VGG16PartialLoss

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train(model, criterion, optimizer, dataset, epoch):
    total_loss = 0
    for i, data in enumerate(dataset):
        optimizer.zero_grad()
        
        masked_img = data['masked_img'].to(device)
        image = data['image'].to(device)
        mask = data['mask'].to(device)

        output, _ = model(image, mask)
        
        loss, vgg_loss, style_loss = criterion(output, image)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i%200 == 0:
            print("Iter[{}] Loss => {:.4}".format(i, loss))
    
    masked_img = masked_img[0].permute((1, 2, 0))
    masked_img = masked_img.cpu().numpy() * 255.0
    image = image[0].permute((1, 2, 0))
    image = image.cpu().numpy() * 255.0
    mask = mask[0].permute((1, 2, 0))
    mask = mask.cpu().numpy()
    cv2.imwrite("masked_img.png", masked_img)
    cv2.imwrite("image.png", image)
    cv2.imwrite("mask.png", mask)
    output = torch.clamp(output, min = 0, max = 1)
    output = output[0].permute((1, 2, 0))
    output = output.detach().cpu().numpy() * 255.0
    cv2.imwrite("output.png", output)
        
    total_loss /= len(dataset)
        
    return total_loss


if __name__ == '__main__':

    args_parser = TrainParser().parser
    args = args_parser.parse_args()
    args = set_init(args)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # create folders
    if not os.path.exists(os.path.join(args.save_dir, args.name)):
        os.makedirs(os.path.join(args.save_dir, args.name))
        
    print_options(args)

    # dataset
    trainset, valset = CreateDataloader(args)
    
    # model
    model = CreateModel(args)
    model.to(device)
    model.train()
    
    # loss
    criterion = VGG16PartialLoss(device)
    
    # optimizer
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = ReduceLROnPlateau(
            optimizer, 'min', patience=3,
            min_lr=1e-10, verbose=True
        )
    
    loss_file = open(os.path.join(args.save_dir, args.name, "loss.txt"), 'w')
    for epoch in range(args.epochs):
        print('\nEpoch %s' % (epoch+1))

        total_loss = train(model, criterion, optimizer, trainset, epoch)
        scheduler.step(total_loss)
        
        print("Epoch[{}] ====> Loss: {:.4}, lr: {:.4}".format(epoch+1, total_loss, optimizer.param_groups[0]['lr']))
        loss_file.write("Epoch[{}] ====> {:.4}\n".format(epoch+1, total_loss))
        
        if epoch%args.save_interval == 0 or epoch == args.epochs-1:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                }, '%s/%s/%s.pt' % (args.save_dir, args.name, epoch+1))
    loss_file.close()
