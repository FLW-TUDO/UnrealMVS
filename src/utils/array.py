import torch
import numpy as np


def sqrt(x):
    return torch.sqrt(x) if isinstance(x, torch.Tensor) else np.sqrt(x)

def atan2(y, x):
    return torch.atan2(y, x) if isinstance(x, torch.Tensor) else np.arctan2(y, x)

def asin(x):
    return torch.asin(x) if isinstance(x, torch.Tensor) else np.arcsin(x)

def acos(x):
    return torch.acos(x) if isinstance(x, torch.Tensor) else np.arccos(x)

def toNumpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

def concat(arr_list: list, axis=0):
    if isinstance(arr_list[0], torch.Tensor):
        return torch.cat(arr_list, dim=axis)
    return np.concatenate(arr_list, axis=axis)

def cast_precision(tensors: list, precision: str) -> list:
    if precision == "float16":
        return [t.half() for t in tensors]
    if precision == "bfloat16":
        return [t.bfloat16() for t in tensors]
    return [t.float() for t in tensors]


def polyval(P, x):
    if isinstance(x, torch.Tensor):
        P = torch.Tensor(P).to(x.device)
    if isinstance(P, torch.Tensor):
        val = torch.zeros_like(x)
        for i in range(len(P) - 1):
            val = val * x + P[i] * x
        val += P[-1]
        return val
    return np.polyval(P, x)
