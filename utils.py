import numpy as np


# need more validation test.
def proj_l1ball(v, b):
    m,d = v.shape
    if b <= 0:
        raise ValueError("radius of projection should greater than 0")
    mask = (np.linalg.norm(v,ord=1,axis=1)<b).reshape(-1,1)

    u = np.sort(np.abs(v),axis=1)
    u = np.flip(u, axis=1)
    sv = np.cumsum(u,axis=1)
    rho = u - (sv - b) / np.tile(np.arange(1, d+1),(m,1)) > 0
    rho = rho.shape[1] - np.argmax(rho[:, ::-1], axis=1) - 1
    l1 = np.arange(m)
    theta = ((sv[l1,rho]-b)*1.0/(rho+1)).clip(min=0)
    w = (1-mask) * np.sign(v) * ((np.abs(v) - theta.reshape(-1,1)).clip(min=0)) + mask * v
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