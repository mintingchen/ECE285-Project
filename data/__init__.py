from data.paris import ParisTrain, ParisTest
from data.cifar import CifarTrain, CifarTest
from torch.utils.data import Dataset, DataLoader

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

    
def CreateDataloader(args, mode='train'):
    assert args.dataset in ['cifar', 'paris']
    if args.dataset == 'paris':
        if mode == 'train':
            train_loader = ParisTrain(args.image_dir, args.image_list_train, args.mask_dir, args.mask_list_train)
            val_loader = ParisTest(args.image_dir, args.image_list_test, args.mask_dir, args.mask_list_train)
            trainset = DataLoader(train_loader, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
            valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
            return trainset, valset
        else:
            val_loader = ParisTest(args.image_dir, args.image_list_test, args.mask_dir, args.mask_list_train)
            valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
            return valset
    
    elif args.dataset == 'cifar':
        if mode == 'train':
            train_loader = CifarTrain(args.image_dir, args.image_list_train, args.mask_dir, args.mask_list_train)
            val_loader = CifarTest(args.image_dir, args.image_list_test, args.mask_dir, args.mask_list_train)
            trainset = DataLoader(train_loader, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
            valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
            return trainset, valset
        else:
            val_loader = CifarTest(args.image_dir, args.image_list_test, args.mask_dir, args.mask_list_train)
            valset = DataLoader(val_loader, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
            return valset