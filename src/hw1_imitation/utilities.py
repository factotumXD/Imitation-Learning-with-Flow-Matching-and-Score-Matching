import torch
from torch.distributions import Beta, Normal

def rand_0_to_09(size,device: torch.device = None):
    return torch.rand(size,device=device) * 0.9

def rand_01_steps_inclusive(size,device: torch.device = None):
    return torch.randint(0, 10, size, dtype=torch.float32,device=device) / 10.0

def sample_t_beta(
    size,
    alpha=1.5,
    beta=1.0,
    device=None,
):
    dist = Beta(alpha, beta)
    x = dist.sample(size)   # X ~ Beta(alpha, beta)
    t = 1.0 - x                                 # t = 1−X
    if device is not None:
        t = t.to(device=device)
    return t

def sample_logit_normal(size,loc=0.0,scale=0.5,device: torch.device = None):
    dist = Normal(loc, scale)
    x = dist.sample(size)
    y = torch.sigmoid(x)
    if device is not None:
        y = y.to(device=device)
    return y   
