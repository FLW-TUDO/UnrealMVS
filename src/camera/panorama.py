"""
Depth-guided fisheye → equirectangular projection.

Pure numpy / cv2 — no torch dependency.
Used by dataset converters and the visualization tool.

Pipeline per frame:
    pts_rig   = depth * rays
    local_pts = rig2cam @ pts_rig          (rig → camera frame)
    col, row  = cam.ray_to_pixel(local_pts)
    pano      = cv2.remap(fisheye, col, row)
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.camera.model.camera import CameraModel
from src.camera.parameter import create_camera_model
import src.utils.geometry as geometry


# ── Ray generation ─────────────────────────────────────────────────────────────

def spherical_rays(equirect_size: tuple, phi_deg: float) -> np.ndarray:
    """
    Return (3, H*W) unit direction vectors for an equirectangular grid.

    Args:
        equirect_size: (H, W) output resolution.
        phi_deg:       vertical half-angle in degrees.
                       45° → ±45° (90° total FOV, OmniMVS training default).
                       90° → ±90° (full hemisphere, House/Things panoramas).
    """
    h, w = equirect_size
    xs, ys = np.meshgrid(range(w), range(h))
    w_2, h_2 = w / 2.0, (h - 1) / 2.0
    xs = (xs - w_2) / w_2 * np.pi + np.pi / 2.0
    ys = (ys - h_2) / h_2 * np.deg2rad(phi_deg)
    x = -np.cos(ys) * np.cos(xs)
    y =  np.sin(ys)
    z =  np.cos(ys) * np.sin(xs)
    return np.concatenate(
        (x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1))
    ).astype(np.float64)


# ── Camera model construction ──────────────────────────────────────────────────

def build_camera_models(intrinsics: list, extrinsics: list) -> List[CameraModel]:
    """
    Build CameraModel objects from calibration JSON entries.

    Accepts the 'intrinsics' and 'extrinsics' lists from inside 'value0'
    of calibration_intrinsic.json / calibration_extrinsic.json.
    """
    cameras: List[CameraModel] = []
    for intr_entry in intrinsics:
        cam = create_camera_model(
            intr_entry['camera_type'],
            intr_entry['camera_name'],
            intr_entry['intrinsics'],
        )
        cameras.append(cam)
    for idx, extr_entry in enumerate(extrinsics):
        if cameras[idx].name == extr_entry['camera_name']:
            cameras[idx].set_extrinsic(extr_entry['extrinsics'])
    return cameras


# ── Projection helpers ─────────────────────────────────────────────────────────

def crop_to_size(depth: np.ndarray, target: tuple) -> np.ndarray:
    """Centre-crop depth rows to target height; resize width only if needed."""
    h, w = target
    if depth.shape == (h, w):
        return depth
    if depth.shape[0] > h:
        sh    = (depth.shape[0] - h) // 2
        depth = depth[sh : sh + h, :]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


def project_frame(
    frame_dir: Path,
    camera_models: List[CameraModel],
    rays: np.ndarray,
    equirect_size: tuple,
) -> tuple:
    """
    Project all fisheye images into the equirectangular grid using GT depth.

    Returns:
        pano       float32 (H, W, 3)  blended panorama.
        cam_valids list of bool (H, W) masks — one per camera.
    """
    h, w = equirect_size

    depth_path = frame_dir / "Equirectangular_Depth.exr"
    depth_raw  = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is not None:
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth_raw  = crop_to_size(depth_raw, equirect_size)
        depth_flat = np.where(depth_raw > 0, depth_raw, 1e4).reshape(1, -1).astype(np.float64)
    else:
        depth_flat = np.full((1, h * w), 1e4, dtype=np.float64)

    pts_rig    = rays * depth_flat
    pano_sum   = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w),    dtype=np.float64)
    cam_valids = []

    for cam_idx, cam in enumerate(camera_models):
        img_path = frame_dir / f"FisheyeCamera_{cam_idx + 1}.exr"
        img      = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            cam_valids.append(np.zeros((h, w), dtype=bool))
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        local_pts = geometry.applyTransform(cam.rig2cam, pts_rig)
        # OcamModel.ray_to_pixel returns shape (2, N): out[0]=col, out[1]=row
        pixel = cam.ray_to_pixel(local_pts)
        col   = pixel[0].reshape(h, w).astype(np.float32)
        row   = pixel[1].reshape(h, w).astype(np.float32)

        ih, iw = cam.resolution[0], cam.resolution[1]
        valid  = (col >= 0) & (row >= 0) & (col < iw) & (row < ih)

        remapped    = cv2.remap(img, col, row,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask        = valid.astype(np.float64)
        pano_sum   += remapped * mask[:, :, np.newaxis]
        weight_sum += mask
        cam_valids.append(valid)

    denom = np.where(weight_sum > 0, weight_sum, 1.0)
    pano  = (pano_sum / denom[:, :, np.newaxis]).astype(np.float32)
    return pano, cam_valids


def save_panorama(
    frame_dir: Path,
    camera_models: List[CameraModel],
    rays: np.ndarray,
    equirect_size: tuple,
) -> None:
    """Project and save Equirectangular.exr for one frame."""
    pano, _ = project_frame(frame_dir, camera_models, rays, equirect_size)
    cv2.imwrite(str(frame_dir / "Equirectangular.exr"), pano)
