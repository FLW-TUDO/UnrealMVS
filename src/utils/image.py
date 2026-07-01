import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.array import *


def interp2D(I, grid):
    istensor = isinstance(I, torch.Tensor)
    I    = torch.tensor(I).float().squeeze().unsqueeze(0)
    grid = torch.Tensor(grid).squeeze().unsqueeze(0)
    if len(I.shape) < 4:
        I = I.unsqueeze(0)
    out = F.grid_sample(I, grid, mode='nearest', align_corners=True).squeeze()
    if not istensor:
        out = out.numpy()
    return out


def pixelToGrid(pts, target_resolution: tuple, source_resolution: tuple) -> np.ndarray:
    h, w       = target_resolution
    height, width = source_resolution
    xs = (pts[0, :]) / (width  - 1) * 2 - 1
    ys = (pts[1, :]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return concat((xs, ys), 2)


def normalizeImage(image: np.ndarray, mask=None, channel_wise_mean=True) -> np.ndarray:
    image = toNumpy(image)

    def _normalize1d(img, m):
        img = img.squeeze().astype(np.float32)
        if m is not None:
            img[~m] = np.nan
        img = (img - np.nanmean(img)) / np.nanstd(img)
        if m is not None:
            img[~m] = 0
        return img

    if len(image.shape) == 3 and image.shape[2] == 3:
        if channel_wise_mean:
            return np.concatenate(
                [_normalize1d(image[:, :, i], mask)[..., np.newaxis] for i in range(3)],
                axis=2,
            )
        img = image.squeeze().astype(np.float32)
        m   = np.tile(mask[..., np.newaxis], (1, 1, 3)) if mask is not None else None
        if m is not None:
            img[~m] = np.nan
        img = (img - np.nanmean(img)) / np.nanstd(img)
        if m is not None:
            img[~m] = 0
        return img
    return _normalize1d(image, mask)
