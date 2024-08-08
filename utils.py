import numpy as np
import torch

def proj_l1ball(v, b):
    device = v.device
    m, d = v.shape
    if b <= 0:
        raise ValueError("radius of projection should be greater than 0")
    
    mask = (torch.norm(v, p=1, dim=1) < b).reshape(-1, 1).to(torch.float)

    u, _ = torch.sort(torch.abs(v), dim=1, descending=True)
    sv = torch.cumsum(u, dim=1)
    rho = (u - (sv - b) / torch.arange(1, d + 1, device='cuda')).gt(0).type(torch.int64)
    rho = d - torch.argmax(rho.flip(dims=[1]), dim=1) - 1
    l1 = torch.arange(m).to(device)
    
    theta = ((sv[l1, rho] - b) / (rho + 1).type(torch.float)).clamp(min=0)
    w = (1 - mask) * torch.sign(v) * (torch.abs(v) - theta.reshape(-1, 1)).clamp(min=0) + mask * v
    return w

class MaxIterError():
    def __init__(self):
        pass


class ConstBatchSizeGenerator(object):
    def __init__(self, batch_size, max_iter) -> None:
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.current_iter = 0
        
    def __call__(self):
        if self.current_iter == self.max_iter:
            raise StopIteration
        else:
            return self.batch_size 
    
    def __len__(self):
        return self.max_iter
