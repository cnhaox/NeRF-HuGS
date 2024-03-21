from typing import Optional
from dataclasses import asdict
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

from utils.utils import State

def load_snapshot(ckpt_file: str, 
                  model=None, 
                  optimizer=None, 
                  scheduler=None, 
                  scaler=None, 
                  device: str='cpu') -> State:
    
    map_location = {'cuda:0' : device}
    ckpt_dict = torch.load(ckpt_file, map_location=map_location)
    if model is not None: model.load_state_dict(ckpt_dict['model'])
    if optimizer is not None: optimizer.load_state_dict(ckpt_dict['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(ckpt_dict['scheduler'])
    if scaler is not None: scaler.load_state_dict(ckpt_dict['scaler'])
    state = State(**ckpt_dict['state'])
    return state


def save_snapshot(ckpt_file: str,
                  state: State,
                  model=None,
                  optimizer=None,
                  scheduler=None,
                  scaler=None):
    state_dict = {
        'state': asdict(state),
        'model': None,
        'optimizer': None,
        'scheduler': None,
        'scaler': None
    }
    if model is not None:
        if isinstance(model, DDP) or isinstance(model, DP):
            state_dict['model'] = model.module.state_dict()
        else: 
            state_dict['model'] = model.state_dict()
    if optimizer is not None: state_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None: state_dict['scheduler'] = scheduler.state_dict()
    if scaler is not None: state_dict['scaler'] = scaler.state_dict()
    torch.save(state_dict, ckpt_file)


def load_weights(ckpt_file: str, model, device: str) -> State:
    map_location = {'cuda:0' : device}
    ckpt_dict = torch.load(ckpt_file, map_location=map_location)
    model.load_state_dict(ckpt_dict['model'])
    state = State(**ckpt_dict['state'])
    return state


def save_weights(ckpt_file: str, state: State, model):
    state_dict = {
        'state': asdict(state),
        'model': None
    }
    state_dict['model'] = \
        model.module.state_dict() if isinstance(model, DDP) or isinstance(model, DP) \
        else model.state_dict()
    torch.save(state_dict, ckpt_file)