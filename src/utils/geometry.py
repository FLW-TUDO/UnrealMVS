import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.array import *


def euler(r: np.ndarray) -> np.ndarray:
    if r.size == 3:
        return R.from_euler('xyz', r.squeeze()).as_matrix()
    return R.from_matrix(r).as_euler('xyz').reshape((3, 1))

def quat(r: np.ndarray) -> np.ndarray:
    if r.size == 4:
        return R.from_quat(r.squeeze()).as_matrix()
    return R.from_matrix(r).as_quat().reshape((3, 1))

def rodrigues(r: np.ndarray) -> np.ndarray:
    if r.size == 3:
        return R.from_rotvec(r.squeeze()).as_matrix()
    if r.size == 4:
        return R.from_quat(r.squeeze()).as_matrix()
    return R.from_matrix(r).as_rotvec().reshape((3, 1))


def getRot(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        return rodrigues(transform.reshape((6, 1))[:3])
    if transform.size == 7:
        return quat(transform.reshape((7, 1))[:4])
    if transform.shape in ((3, 4), (4, 4)):
        return transform[:3, :3]
    return None

def getTr(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        return transform.reshape((6, 1))[3:6].reshape((3, 1))
    if transform.size == 7:
        return transform.reshape((7, 1))[4:7].reshape((3, 1))
    if transform.shape in ((3, 4), (4, 4)):
        return transform[:3, 3].reshape((3, 1))
    return None


def inverseTransform(transform: np.ndarray) -> np.ndarray:
    rot, tr  = getRot(transform), getTr(transform)
    R_inv    = rot.T
    tr_inv   = -R_inv.dot(tr)
    if transform.size in (6, 7):
        r_inv = rodrigues(R_inv)
        return np.concatenate((r_inv, tr_inv), axis=0)
    return np.concatenate((R_inv, tr_inv), axis=1)


def applyTransform(transform: np.ndarray, P):
    rot, tr = getRot(transform), getTr(transform)
    if isinstance(P, torch.Tensor):
        rot = torch.Tensor(rot).to(P.device)
        tr  = torch.Tensor(tr).to(P.device)
        return torch.matmul(rot, P) + tr
    return rot.dot(P) + tr
