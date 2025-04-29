import torch
import numpy as np
from PIL import Image

def get_gradient(w_global_list, w_client_list, lr):
    grad = []
    for layer in w_global_list:
        if w_global_list[layer].requires_grad:
            grad.append((w_global_list[layer].detach().clone() - w_client_list[layer].detach().clone()) / lr)
    return grad

def get_x_init(args, dm, ds, device):
    if args.init_path is not None:
        print('loading x_init from {}'.format(args.init_path))
        x_init = torch.as_tensor(np.array(Image.open(args.init_path).resize((256, 256), Image.BICUBIC)) / 255).to(device)
        x_init = x_init.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).float()
        return x_init
    else:
        NotImplementedError()
    return

