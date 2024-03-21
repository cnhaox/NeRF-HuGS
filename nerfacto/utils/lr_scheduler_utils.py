import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_decay_scheduler(optimizer: Optimizer,
                               lr_init: float,
                               lr_final: float,
                               lr_delay_mult: float,
                               warmup_steps: int,
                               max_steps: int):
    # The step in LRScheduler begins from 0
    # After initialization, LRScheduler will automatically execute scheduler.step(0),
    # so next scheduler.step() should be used after optimizer.step()
    def func(step):
        if step < warmup_steps:
            factor = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / warmup_steps, 0, 1)
            )
        else:
            t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            factor = lr / lr_init
        
        return factor
    
    scheduler = LambdaLR(optimizer, lr_lambda=func)
    return scheduler