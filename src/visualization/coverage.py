"""
Camera coverage overview visualizer.

Generates a PNG with one row per fisheye camera (projected into equirectangular
space), a combined panorama row, and a GT depth colormap row.

Usage:
    from src.visualization.coverage import visualize_frame
    visualize_frame("datasets/Urban_Sunny", "0001")
"""

import json
from pathlib import Path

import cv2
import numpy as np

import src.utils.geometry as geometry_utils
from src.camera.panorama import (
    build_camera_models,
    crop_to_size,
    project_frame,
    spherical_rays,
)

_CAMERA_COLORS = [
    (0.9, 0.3, 0.3),   # Cam 1 — red
    (0.3, 0.8, 0.3),   # Cam 2 — green
    (0.3, 0.5, 1.0),   # Cam 3 — blue
    (1.0, 0.8, 0.2),   # Cam 4 — yellow
]


def _to_8bit(img: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    img = np.clip(img, 0, 1)
    return (np.power(img, gamma) * 255).astype(np.uint8)


def _labeled_row(img_8bit: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((20, img_8bit.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.vstack([bar, img_8bit])


def _depth_colormap(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    d_min = depth[valid].min()  if valid.any() else 0.0
    d_max = np.percentile(depth[valid], 95) if valid.any() else 1.0
    norm  = np.zeros_like(depth)
    norm[valid] = np.clip(1.0 - (depth[valid] - d_min) / (d_max - d_min + 1e-6), 0, 1)
    cmap  = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cmap[~valid] = 20
    return cmap


def visualize_frame(
    dataset_path: str | Path,
    frame_name: str,
    equirect_size: tuple[int, int] | None = None,
    phi_deg: float | None = None,
) -> str:
    """
    Generate a PNG overview for one frame and save it next to the frame dir.

    If equirect_size / phi_deg are not given, they are read from the dataset's
    calibration JSON (falls back to 160×640 / 45° if not present).

    Returns the output path.
    """
    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    dataset_path = Path(dataset_path)
    frame_dir    = dataset_path / frame_name

    intrinsic = json.loads((dataset_path / "calibration_intrinsic.json").read_text())
    extrinsic = json.loads((dataset_path / "calibration_extrinsic.json").read_text())
    camera_models = build_camera_models(
        intrinsic["value0"]["intrinsics"],
        extrinsic["value0"]["extrinsics"],
    )

    if equirect_size is None:
        equirect_size = (160, 640)
    if phi_deg is None:
        phi_deg = 45.0

    rays = spherical_rays(equirect_size, phi_deg)
    h, w = equirect_size

    depth_raw = cv2.imread(
        str(frame_dir / "Equirectangular_Depth.exr"), cv2.IMREAD_UNCHANGED
    )
    if depth_raw is None:
        raise RuntimeError(f"Depth not found: {frame_dir / 'Equirectangular_Depth.exr'}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    depth_raw = crop_to_size(depth_raw, equirect_size)

    pano, cam_valids = project_frame(frame_dir, camera_models, rays, equirect_size)
    panels = []

    depth_flat = np.where(depth_raw > 0, depth_raw, 1e4).reshape(1, -1).astype(np.float64)
    pts_rig    = rays * depth_flat

    for cam_id, (valid, cam) in enumerate(zip(cam_valids, camera_models), 1):
        img = cv2.imread(str(frame_dir / f"FisheyeCamera_{cam_id}.exr"), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        local_pts = geometry_utils.applyTransform(cam.rig2cam, pts_rig)
        pixel     = cam.ray_to_pixel(local_pts)   # out[0]=col, out[1]=row
        map_col   = pixel[0].reshape(h, w).astype(np.float32)
        map_row   = pixel[1].reshape(h, w).astype(np.float32)
        remapped = cv2.remap(img, map_col, map_row,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        remapped[~valid] = 0.0

        tinted  = remapped.copy()
        color   = _CAMERA_COLORS[(cam_id - 1) % len(_CAMERA_COLORS)]
        overlay = np.zeros_like(tinted)
        overlay[~valid] = [c * 0.15 for c in color]
        tinted  = np.clip(tinted + overlay, 0, 1)

        coverage = valid.mean() * 100
        panels.append(_labeled_row(_to_8bit(tinted), f"Camera {cam_id}  ({coverage:.0f}% coverage)"))
        print(f"  Camera {cam_id}: {coverage:.1f}% coverage")

    panels.append(_labeled_row(_to_8bit(pano), "Panorama (all cameras combined)"))
    panels.append(_labeled_row(_depth_colormap(depth_raw),
                               "GT Depth  (INFERNO: bright = close,  dark = far)"))

    out_path = str(dataset_path / f"{frame_name}_overview.png")
    cv2.imwrite(out_path, np.vstack(panels))
    print(f"  Saved → {out_path}")
    return out_path
