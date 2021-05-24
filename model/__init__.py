import torch

from model.model import Unet, Unet_light

def CreateModel(args):
    if args.model == "Unet":
        model = Unet()
    else:
        model = Unet_light()
    
    return model