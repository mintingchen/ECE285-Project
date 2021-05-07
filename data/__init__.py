from data.paris import ParisTrain
from torch.utils.data import Dataset, DataLoader

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

    
def CreateDataloader(args, mode='train'):
    if mode == 'train':
        data_loader = ParisTrain(args.image_dir, args.image_list, args.mask_dir)
        dataset = DataLoader(data_loader, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
        
        return dataset
    else:
        pass