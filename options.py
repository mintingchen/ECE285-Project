import argparse
import torch
import os
import numpy as np
import random

def set_seeds(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for replicable results this can be activated
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Inpainting Training')
        # Frequently used commands
        parser.add_argument('--epochs', type=int, default=1, help='')
        parser.add_argument('--batch_size', type=int, default=16, help='')
        parser.add_argument('--image_dir', type=str, default='dataset/paris/', help='')
        parser.add_argument('--image_list_train', type=str, default='namelist/paris_training.txt', help='')
        parser.add_argument('--image_list_test', type=str, default='namelist/paris_training.txt', help='')
        parser.add_argument('--mask_dir', type=str, default='dataset/mask/testing_mask_dataset/', help='')
        parser.add_argument('--mask_list_train', type=str, default='namelist/nv_mask_training.txt', help='')
        parser.add_argument('--mask_list_test', type=str, default='namelist/nv_mask_training.txt', help='')

        self.parser = parser


class TestParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Inpainting Testing')
        # Frequently used commands
        parser.add_argument('--model', type=str, help='Model type')
        parser.add_argument('--task', type=str, help='Task type: denoising_SR, denoising, SR')
        parser.add_argument('--dataset', type=str, help='Dataset type: unity, ue, ueGB')

def print_options(args):
        message = ''
        message = '------------------------ Options ------------------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------------- End -------------------------'
        print(message)

#         file = open(os.path.join(args.save_dir, args.name, 'configs'), 'w')
#         file.write(message)
#         file.close()

def set_seeds(opt_in, is_train=True):
    set_seeds(1234)
    opt = opt_in

    return opt
