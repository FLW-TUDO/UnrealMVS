import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def read_depth(depth_path: str) -> np.ndarray:
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def fix_depth_values(depth: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Fill zero pixels with the mean of their non-zero neighbourhood."""
    fixed    = depth.copy()
    zeros    = np.where(depth == 0)
    pad      = kernel_size // 2
    padded   = cv2.copyMakeBorder(depth, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    for y, x in zip(zeros[0], zeros[1]):
        neighbourhood = padded[y : y + kernel_size, x : x + kernel_size]
        valid = neighbourhood[neighbourhood != 0]
        if len(valid) > 0:
            fixed[y, x] = np.mean(valid)
    return fixed


def write_depth(depth: np.ndarray, path: str) -> None:
    if depth.ndim != 2:
        raise ValueError("Depth array must be 2D")
    depth_3d = depth.astype(np.float32).reshape(depth.shape[0], depth.shape[1], 1)
    if not cv2.imwrite(path, depth_3d):
        raise IOError(f"Failed to write depth image to {path}")


def index_to_invdepth(
    idx,
    min_invdepth: float,
    sample_step_invdepth: float,
    start_index: float = 0,
):
    """Convert inverse-depth index to actual inverse-depth value."""
    return min_invdepth + (idx - start_index) * sample_step_invdepth
