from data.paris import ParisTrain, ParisTest
from data.cifar import CifarTrain, CifarTest
from torch.utils.data import Dataset, DataLoader

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

    
def CreateDataloader(args, mode='train', dataset='cifar'):
    if mode == 'train' and dataset == 'paris':
        train_loader = ParisTrain(args.image_dir, args.image_list_train, args.mask_dir)
        val_loader = ParisTest(args.image_dir, args.image_list_test, args.mask_dir)
        trainset = DataLoader(train_loader, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
        valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
        return trainset, valset
    
    elif mode != 'train' and dataset == 'paris':
        val_loader = ParisTest(args.image_dir, args.image_list_test, args.mask_dir)
        valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
        return valset
    
    elif mode == 'train' and dataset == 'cifar':
        train_loader = CifarTrain(args.image_dir, args.image_list_train, args.mask_dir)
        val_loader = CifarTest(args.image_dir, args.image_list_test, args.mask_dir)
        trainset = DataLoader(train_loader, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
        valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
        return trainset, valset
    
    elif mode != 'train' and dataset == 'cifar':
        val_loader = CifarTest(args.image_dir, args.image_list_test, args.mask_dir)
        valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
        return valset