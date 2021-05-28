import torch
import torch.nn as nn

from model.model import Unet, Unet_light

def CreateModel(args):
    if args.model == "Unet":
        model = Unet()
    else:
        model = Unet_light()
    if args.mode == "ft":
        assert args.init_weight is not None
        ckpt = torch.load(args.init_weight)
        model.load_state_dict(ckpt['state_dict'])
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                print("frozen")
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        print("Model Initialization Finished")
        
    
    return model