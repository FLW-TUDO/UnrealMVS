import numpy as np
import torch


def absolute_error_loss(pred, target, valid):
    error = torch.abs(pred[valid] - torch.round(target[valid]))
    return torch.nanmean(error)


def smooth_l1_loss(pred, target, valid, sigma=1.0):
    x    = torch.abs(pred[valid] - target[valid])
    loss = torch.where(x < sigma, 0.5 * x ** 2, sigma * (x - 0.5 * sigma))
    return torch.nanmean(loss)


def entropy_boundary_loss(x, k, a, reduction='mean'):
    x = torch.relu(a * (x - np.log(k)))
    return torch.mean(x) if reduction == 'mean' else torch.sum(x)


def ent_exploss(x, k, a):
    return torch.relu(a / k * (torch.exp(x) - k))


def ent_huber(x, sigma):
    return torch.where(x < sigma, 0.5 * x ** 2, sigma * (x - 0.5 * sigma))


def ent_huberk(x, sigma, k):
    inv_k    = 1.0 / k
    sigma_k  = sigma ** (k - 1)
    sigma_k2 = (k - 1.0) / k * sigma
    return torch.where(x < sigma, inv_k * x ** k, sigma_k * (x - sigma_k2))
