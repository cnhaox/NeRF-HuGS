from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
import random

import numpy as np
import torch
from torch import Tensor
from torch.types import Device

@dataclass
class State:
    step: int = 0
    epoch: int = 0
    next_eval_idx: int = 0


def init_seed(seed: int=777) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_device(data, device: Device):
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        return [to_device(data_, device) for data_ in data]
    else:
        return data
    

def split_tensor_dict(data: Dict[str, Tensor], chunk_size: int) -> List[Dict[str, Tensor]]:
    keys = list(data.keys())
    total_size = data[keys[0]].shape[0]
    nums = total_size//chunk_size + min(total_size%chunk_size, 1)
    sub_datas = [dict() for _ in range(nums)]
    for key in keys:
        for i, sub_tensor in enumerate(data[key].split(chunk_size, dim=0)):
            sub_datas[i][key] = sub_tensor

    return sub_datas


def merge_tensor_dict(datas: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    output = {}
    for key in datas[0].keys():
        output[key] = [datas[0][key]]
    for i in range(1, len(datas)):
        for key in output.keys(): output[key].append(datas[i][key])
    for key in output.keys():
        output[key] = torch.cat(output[key], dim=0)
    return output


def split_tensor_data(data, chunk_size: int) -> list:
    output = None
    if isinstance(data, Tensor):
        output = torch.split(data, split_size_or_sections=chunk_size, dim=0)
    elif isinstance(data, list):
        for i in range(len(data)):
            sub_data = split_tensor_data(data[i], chunk_size)
            if output is None: 
                output = [list() for _ in range(len(sub_data))]
            for j in range(len(sub_data)):
                output[j].append(sub_data[j])
    elif isinstance(data, dict):
        for key in data.keys():
            sub_data = split_tensor_data(data[key], chunk_size)
            if output is None: 
                output = [dict() for _ in range(len(sub_data))]
            for j in range(len(sub_data)):
                output[j][key] = sub_data[j]
    else:
        raise NotImplementedError()
    
    return output


def merge_tensor_data(datas: list):
    if isinstance(datas[0], Tensor):
        output = torch.cat(datas, dim=0)
    elif isinstance(datas[0], list):
        output = []
        for j in range(len(datas[0])):
            output.append(
                merge_tensor_data([datas[i][j] for i in range(len(datas))])
            )
    elif isinstance(datas[0], dict):
        output = {}
        for key in datas[0].keys():
            output[key] = \
                merge_tensor_data([datas[i][key] for i in range(len(datas))])
    else:
        raise NotImplementedError()
    
    return output