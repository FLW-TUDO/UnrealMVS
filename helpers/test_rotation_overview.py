#!/usr/bin/env python3
"""
Camera overview: each camera individually, combined panorama, GT depth.

Single frame (default):
    python helpers/test_rotation_overview.py --dataset datasets/PolarStation --frame 0001

All frames in a split:
    python helpers/test_rotation_overview.py --dataset datasets/PolarStation --all-frames

Options:
    --dataset    Path to dataset folder  (default: datasets/PolarStation)
    --frame      Frame ID                (default: 0001)
    --split      Split to use for --all-frames: train, eval, test, or flat (default: train)
    --all-frames Process all frames in the split
    --out        Output directory        (default: rotation_test)
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.geometry as geo
from src.camera.panorama import build_camera_models, crop_to_size, spherical_rays
from src.camera.parameter import load_camera_params

EQUIRECT_SIZE = (160, 640)
PHI_DEG       = 45.0

CAMERA_COLORS = [
    (0.9, 0.3, 0.3),
    (0.3, 0.8, 0.3),
    (0.3, 0.5, 1.0),
    (1.0, 0.8, 0.2),
]


def to_8bit(img, gamma=0.45):
    return (np.power(np.clip(img, 0, 1), gamma) * 255).astype(np.uint8)


def labeled_row(img_8bit, text):
    bar = np.zeros((20, img_8bit.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.vstack([bar, img_8bit])


def depth_colormap(depth):
    valid = (depth > 0) & (depth < 1e4)
    norm  = np.zeros_like(depth)
    if valid.any():
        d_min = depth[valid].min()
        d_max = np.percentile(depth[valid], 95)
        norm[valid] = np.clip(1.0 - (depth[valid] - d_min) / (d_max - d_min + 1e-6), 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cmap[~valid] = 20
    return cmap


def process_frame(frame_dir: Path, cameras, rays) -> np.ndarray:
    h, w = EQUIRECT_SIZE

    depth_raw = cv2.imread(str(frame_dir / "Equirectangular_Depth.exr"), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"No depth file in {frame_dir}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    depth_raw  = crop_to_size(depth_raw, EQUIRECT_SIZE)
    depth_flat = np.where((depth_raw > 0) & (depth_raw < 1e4), depth_raw, 1e4)
    pts_rig    = rays * depth_flat.reshape(1, -1).astype(np.float64)

    pano_sum   = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w),    dtype=np.float64)
    panels     = []

    for ci, cam in enumerate(cameras):
        img = cv2.imread(str(frame_dir / f"FisheyeCamera_{ci+1}.exr"), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = img.astype(np.float32)

        local = geo.applyTransform(cam.cam2rig, pts_rig)
        px    = cam.ray_to_pixel(local)
        mc    = px[0].reshape(h, w).astype(np.float32)
        mr    = px[1].reshape(h, w).astype(np.float32)

        ih, iw = cam.resolution[0], cam.resolution[1]
        valid  = (mc >= 0) & (mr >= 0) & (mc < iw) & (mr < ih)

        remapped = cv2.remap(img, mc, mr, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        remapped[~valid] = 0.0

        mask        = valid.astype(np.float64)
        pano_sum   += remapped * mask[:, :, np.newaxis]
        weight_sum += mask

        tinted  = remapped.copy()
        color   = CAMERA_COLORS[ci % len(CAMERA_COLORS)]
        overlay = np.zeros_like(tinted)
        overlay[~valid] = [c * 0.15 for c in color]
        tinted  = np.clip(tinted + overlay, 0, 1)

        panels.append(labeled_row(to_8bit(tinted), f"Camera {ci+1}"))

    denom = np.where(weight_sum > 0, weight_sum, 1.0)
    pano  = (pano_sum / denom[:, :, np.newaxis]).astype(np.float32)
    panels.append(labeled_row(to_8bit(pano), "Panorama (fisheye + depth)"))
    panels.append(labeled_row(depth_colormap(depth_raw), "GT Depth"))

    equirect_path = frame_dir / "Equirectangular.exr"
    orig = cv2.imread(str(equirect_path), cv2.IMREAD_UNCHANGED) if equirect_path.exists() else None
    if orig is not None:
        if orig.ndim == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        if orig.shape[2] > 3:
            orig = orig[:, :, :3]
        orig = crop_to_size(orig.astype(np.float32), EQUIRECT_SIZE)
        panels.append(labeled_row(to_8bit(orig), "Original Equirectangular"))

    return np.vstack(panels)


def load_cameras(dataset_dir: Path):
    return load_camera_params(dataset_dir)


def _frame_dirs_in_split(dataset_dir: Path, split: str) -> list[Path]:
    """Return sorted frame dirs, handling both split and flat layouts."""
    split_dir = dataset_dir / split
    if split_dir.is_dir():
        return sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name.isdigit())
    # flat layout (no split yet)
    return sorted(d for d in dataset_dir.iterdir() if d.is_dir() and d.name.isdigit())


def _worker(args):
    frame_dir, dataset_dir, out_dir = args
    frame_dir   = Path(frame_dir)
    dataset_dir = Path(dataset_dir)
    out_dir     = Path(out_dir)
    cameras = load_cameras(dataset_dir)
    rays    = spherical_rays(EQUIRECT_SIZE, PHI_DEG)
    img = process_frame(frame_dir, cameras, rays)
    out = out_dir / f"overview_{frame_dir.name}.png"
    cv2.imwrite(str(out), img)
    return frame_dir.name, None


def main():
    parser = argparse.ArgumentParser(description="Camera overview for UnrealMVS datasets.")
    parser.add_argument("--dataset",    default="datasets/PolarStation",
                        help="Path to dataset folder (default: datasets/PolarStation).")
    parser.add_argument("--frame",      default="0001",
                        help="Frame ID for single-frame mode (default: 0001).")
    parser.add_argument("--split",      default="train",
                        choices=["train", "eval", "test", "flat"],
                        help="Split to use for --all-frames (default: train).")
    parser.add_argument("--all-frames", action="store_true",
                        help="Process all frames in the split.")
    parser.add_argument("--out",        default="rotation_test",
                        help="Output directory (default: rotation_test).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset not found: {dataset_dir.resolve()}")
        sys.exit(1)

    if args.all_frames:
        frames = _frame_dirs_in_split(dataset_dir, args.split)
        if not frames:
            print(f"No frame directories found in '{dataset_dir}/{args.split}'.")
            return
        workers = min(cpu_count(), len(frames), 8)
        print(f"Processing {len(frames)} frames from {dataset_dir.name}/{args.split} "
              f"({workers} threads)...")
        tasks  = [(str(fd), str(dataset_dir), str(out_dir)) for fd in frames]
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_worker, t): t[0] for t in tasks}
            with tqdm(total=len(frames), unit="frame") as bar:
                for fut in as_completed(futs):
                    try:
                        frame_id, _ = fut.result()
                        bar.set_postfix_str(frame_id)
                    except Exception as e:
                        frame_id = Path(futs[fut]).name
                        bar.set_postfix_str(f"{frame_id} ERROR")
                        errors.append((frame_id, str(e)))
                    bar.update(1)
        print(f"\nDone. {len(frames) - len(errors)}/{len(frames)} succeeded.")
        if errors:
            for frame_id, err in sorted(errors):
                print(f"  {frame_id}: {err}")
    else:
        # Single frame — check both split subdir and flat layout
        split_dir = dataset_dir / args.split
        frame_dir = (split_dir if split_dir.is_dir() else dataset_dir) / args.frame
        if not frame_dir.exists():
            print(f"[ERROR] Frame not found: {frame_dir}")
            sys.exit(1)
        cameras = load_cameras(dataset_dir)
        rays    = spherical_rays(EQUIRECT_SIZE, PHI_DEG)
        print(f"Processing {dataset_dir.name}/{args.frame}...")
        img = process_frame(frame_dir, cameras, rays)
        out = out_dir / f"overview_{dataset_dir.name}_{args.frame}.png"
        cv2.imwrite(str(out), img)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
