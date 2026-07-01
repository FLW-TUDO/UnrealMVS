"""
Core evaluation logic shared by eval_pretrained.py and main.py --evaluate.
"""

import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.datasets.dataset import Dataset
from src.models.neural_networks.omnimvs import OmniMVS
from src.training.error import eval_error
import src.utils.array as array_utils
from src.utils.array import cast_precision
from src.utils.checkpoint import load_state_dict
import src.evaluation.table as table


def prepare_model(model, device, precision):
    model.eval().to(device)
    if precision == "float16":
        return model.half()
    if precision == "bfloat16":
        return model.bfloat16()
    return model.float()


def infer(model, device, images, grids, precision):
    """Run a single forward pass; returns raw model output tensor."""
    imgs = [torch.tensor(img).unsqueeze(0).to(device) for img in images]
    grds = [torch.tensor(grid).to(device) for grid in grids]
    imgs = cast_precision(imgs, precision)
    grds = cast_precision(grds, precision)
    with torch.no_grad():
        return model(torch.stack(imgs), torch.stack(grds))


def save_invdepth(invdepth: np.ndarray, path: str):
    norm = (invdepth - invdepth.min()) / (invdepth.max() - invdepth.min() + 1e-8)
    cv2.imwrite(path, (norm * 255).clip(0, 255).astype(np.uint8))


def evaluate_split(model, device, dataset, precision, save_dir=None):
    """Iterate over all frames in dataset; return averaged metrics dict and skipped count."""
    num_invdepth = dataset.num_invdepth
    n = dataset.data_size
    e1 = e3 = e5 = mae = rmse = valid_ratio = 0.0
    skipped = 0

    for i in tqdm(range(n), leave=False, ncols=100):
        try:
            images, gt_idx, valid, _, grids = dataset.load_sample(i)
        except Exception as exc:
            print(f"  [warn] frame {i} load error: {exc}")
            skipped += 1
            continue

        try:
            pred_idx = infer(model, device, images, grids, precision)
        except Exception as exc:
            print(f"  [warn] frame {i} infer error: {exc}")
            skipped += 1
            continue

        _e1, _e3, _e5, _mae, _rms = eval_error(pred_idx, gt_idx, valid, num_invdepth)
        e1 += _e1; e3 += _e3; e5 += _e5; mae += _mae; rmse += _rms
        valid_ratio += np.sum(valid) / valid.size

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            invd = array_utils.toNumpy(dataset.index_to_invdepth(pred_idx).squeeze())
            save_invdepth(invd, os.path.join(save_dir, f"{i:04d}_pred.png"))

    n_valid = n - skipped
    if n_valid == 0:
        return None, skipped
    return {
        "frames":      n,
        "skipped":     skipped,
        "e1":          e1  / n_valid,
        "e3":          e3  / n_valid,
        "e5":          e5  / n_valid,
        "mae":         mae / n_valid,
        "rmse":        rmse / n_valid,
        "valid_ratio": valid_ratio / n_valid,
    }, skipped


def run_evaluation(config, weights_path, splits, datasets, precision, device,
                   save_dir=None, log_path=None):
    """Load model, evaluate on given datasets/splits, print table, optionally write log.

    Returns: results dict  {ds_name: {split: metrics | None}}
    """
    print(f"Loading model from '{weights_path}' …")
    model = OmniMVS(config)
    model.load_state_dict(load_state_dict(weights_path, device))
    prepare_model(model, device, precision)
    print(f"Device: {device}  Precision: {precision}\n")

    results = {}
    for ds_name in datasets:
        results[ds_name] = {}
        for split in splits:
            try:
                dataset = Dataset(ds_name, config, split=split)
            except FileNotFoundError as e:
                print(f"  [{ds_name}/{split}] skipped: {e}")
                results[ds_name][split] = None
                continue

            save_subdir = os.path.join(save_dir, ds_name, split) if save_dir else None
            metrics, skipped = evaluate_split(
                model, device, dataset, precision, save_dir=save_subdir
            )
            if skipped:
                print(f"  ({skipped} frames skipped due to errors)")
            results[ds_name][split] = metrics

            if metrics:
                m = metrics
                print(
                    f"  [{ds_name}/{split}] {m['frames']} frames  "
                    f"E1: {m['e1']:.2f}  E3: {m['e3']:.2f}  E5: {m['e5']:.2f}  "
                    f"MAE: {m['mae']:.2f}  RMSE: {m['rmse']:.2f}"
                )

    # ── Summary table ──────────────────────────────────────────────────────────
    sep = table.SEP
    print(f"\n{'=' * len(sep)}")
    print("  SUMMARY")
    print(f"{'=' * len(sep)}")
    print(table.hdr())
    print(sep)
    for ds_name in datasets:
        for split in splits:
            m = results[ds_name].get(split)
            frames = m["frames"] if m else 0
            print(table.row(ds_name, split, frames, m))
    print(sep)

    ref_datasets = [d for d in datasets if d in table.PAPER_REF]
    if ref_datasets:
        print(f"\nPaper reference (OmniMVS32 + Lreg+Lent, test split):")
        print(sep)
        print(table.hdr())
        print(sep)
        for ds_name in ref_datasets:
            print(table.ref_row(ds_name, table.PAPER_REF[ds_name]))
        print(sep)

    # ── Write log ──────────────────────────────────────────────────────────────
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"weights:   {weights_path}\n")
            f.write(f"device:    {device}\n")
            f.write(f"precision: {precision}\n\n")
            f.write(table.hdr() + "\n")
            f.write(sep + "\n")
            for ds_name in datasets:
                for split in splits:
                    m = results[ds_name].get(split)
                    frames = m["frames"] if m else 0
                    f.write(table.row(ds_name, split, frames, m) + "\n")
            f.write(sep + "\n")
            if ref_datasets:
                f.write("\nPaper reference (OmniMVS32 + Lreg+Lent, test split):\n")
                f.write(sep + "\n")
                f.write(table.hdr() + "\n")
                f.write(sep + "\n")
                for ds_name in ref_datasets:
                    f.write(table.ref_row(ds_name, table.PAPER_REF[ds_name]) + "\n")
                f.write(sep + "\n")
        print(f"\nSaved to {log_path}")

    return results
