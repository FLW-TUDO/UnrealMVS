"""
UnrealMVS dataset setup — download, convert, split, visualize.

Subcommands:
  setup     Download, extract and convert all (or one) dataset(s)
  convert   Convert already-downloaded raw data
  split     Assign converted frames to train/eval/test subdirectories
  visualize Generate a camera-coverage overview PNG for one frame

Usage:
  python setup_datasets.py setup     [--raw RAW] [--out OUT] [--dataset NAME]
                                     [--panorama] [--visualize]
  python setup_datasets.py convert   [--raw RAW] [--out OUT] [--dataset NAME]
                                     [--panorama] [--panorama-only]
  python setup_datasets.py split     [--out OUT] [--dataset NAME] [--seed SEED]
  python setup_datasets.py verify    [--out OUT] [--dataset NAME] [--workers N]
  python setup_datasets.py visualize --dataset DATASET [--frame FRAME]

Steps and skip logic:
  1. Download      — skipped if file already exists
  2. Extract       — skipped if .extracted marker exists
  3. Convert       — skipped per frame if all EXR outputs already exist
  4. Split         — skipped per dataset if train/ subdir already exists
  5. Panorama      — only with --panorama; skipped per frame if Equirectangular.exr exists
  6. Overview PNG  — only with --visualize; skipped if <frame>_overview.png exists
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.datasets.registry import DATASETS
from src.datasets.download import download_and_extract
from src.datasets.converters import CONVERTERS
from src.visualization.coverage import visualize_frame

_EQUIRECT_DEFAULT = (160, 640)
_PHI_DEFAULT      = 45.0


def _get_converter_params(dataset: dict) -> tuple:
    equirect_size = tuple(dataset.get("equirect_size", _EQUIRECT_DEFAULT))
    phi_deg       = float(dataset.get("phi_deg", _PHI_DEFAULT))
    return equirect_size, phi_deg


def cmd_setup(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw)
    out_dir  = Path(args.out)

    selected = DATASETS
    if args.dataset:
        selected = [d for d in DATASETS if args.dataset.lower() in d["name"].lower()]
        if not selected:
            print(f"No dataset matching '{args.dataset}'.")
            sys.exit(1)

    print(f"Output  : {out_dir.resolve()}")
    print(f"Raw data: {raw_root.resolve()}")

    for dataset in selected:
        print(f"\n{'─' * 60}")
        print(f"  {dataset['name']}")
        print(f"{'─' * 60}")

        download_and_extract(dataset, raw_root)

        converter_key = dataset.get("convert")
        if not converter_key:
            continue

        converter = CONVERTERS.get(converter_key)
        if converter is None:
            print(f"  [error]    No converter for key '{converter_key}'")
            continue

        equirect_size, phi_deg = _get_converter_params(dataset)
        print(f"\n  [convert]  Running converter …")
        converter(
            raw_dir           = raw_root / dataset["raw_dir"],
            out_dir           = out_dir,
            workers           = args.workers,
            equirect_size     = equirect_size,
            phi_deg           = phi_deg,
            generate_panorama = args.panorama,
        )

        if args.visualize:
            for vis_name in dataset.get("output_dirs", []):
                vis_dir = out_dir / vis_name
                if not vis_dir.exists():
                    continue
                frames = sorted(
                    d.name for d in vis_dir.iterdir() if d.is_dir() and d.name.isdigit()
                )
                if not frames:
                    continue
                out_png = vis_dir / f"{frames[0]}_overview.png"
                if out_png.exists():
                    print(f"  [skip]     Overview already exists: {out_png.name}")
                    continue
                print(f"\n  [visualize] {vis_name}/{frames[0]} …")
                visualize_frame(vis_dir, frames[0], equirect_size, phi_deg)

    print(f"\n{'─' * 60}")
    print("  Conversion done.")
    print(f"{'─' * 60}\n")

    if args.split:
        cmd_split(args)

    if args.verify:
        cmd_verify(args)


def cmd_convert(args: argparse.Namespace) -> None:
    selected = DATASETS
    if args.dataset:
        selected = [d for d in DATASETS if args.dataset.lower() in d["name"].lower()]
        if not selected:
            print(f"No dataset matching '{args.dataset}'.")
            sys.exit(1)

    for dataset in selected:
        converter_key = dataset.get("convert")
        if not converter_key:
            continue
        converter = CONVERTERS.get(converter_key)
        if converter is None:
            print(f"[error] No converter for key '{converter_key}'")
            continue
        equirect_size, phi_deg = _get_converter_params(dataset)
        converter(
            raw_dir           = Path(args.raw) / dataset["raw_dir"],
            out_dir           = Path(args.out),
            workers           = args.workers,
            equirect_size     = equirect_size,
            phi_deg           = phi_deg,
            generate_panorama = args.panorama,
            panorama_only     = args.panorama_only,
        )


_SPLIT_DIRS = ("train", "eval", "test")


def _frame_dirs(path: Path) -> list:
    return sorted(d for d in path.iterdir() if d.is_dir() and d.name.isdigit())


_REQUIRED_EXRS = [
    "FisheyeCamera_1.exr",
    "FisheyeCamera_2.exr",
    "FisheyeCamera_3.exr",
    "FisheyeCamera_4.exr",
    "Equirectangular_Depth.exr",
    "Equirectangular.exr",
]


def _verify_frame(frame_dir: Path) -> list:
    """Check all required EXRs in one frame dir. Returns [(exr_name, 'missing'|'corrupt')]."""
    errors = []
    for exr in _REQUIRED_EXRS:
        p = frame_dir / exr
        if not p.exists():
            errors.append((exr, "missing"))
        else:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors.append((exr, "corrupt"))
    return errors


def cmd_split(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    seed    = args.seed

    selected = DATASETS
    if args.dataset:
        selected = [d for d in DATASETS if args.dataset.lower() in d["name"].lower()]
        if not selected:
            print(f"No dataset matching '{args.dataset}'.")
            sys.exit(1)

    print(f"Output: {out_dir.resolve()}")
    print(f"Seed:   {seed}\n")

    for dataset in selected:
        split_cfg = dataset.get("split")
        if not split_cfg:
            print(f"  [{dataset['name']}] No split config — skipping")
            continue

        for output_dir_name in dataset.get("output_dirs", []):
            ds_dir = out_dir / output_dir_name
            if not ds_dir.exists():
                print(f"  [skip]  {output_dir_name} — not found at {ds_dir}")
                continue

            if (ds_dir / "train").exists():
                print(f"  [skip]  {output_dir_name} — already split")
                continue

            frame_dirs = _frame_dirs(ds_dir)
            n_total = len(frame_dirs)
            n_test  = split_cfg["test"]
            n_remainder = n_total - n_test
            n_train = n_remainder // 2
            n_eval  = n_remainder - n_train

            if n_total < n_test + 2:
                print(f"  [error] {output_dir_name} — {n_total} frames found, "
                      f"need at least {n_test + 2}")
                continue

            rng  = np.random.default_rng(seed)
            perm = rng.permutation(n_total)

            # Sort within each split so frame dirs are in numerical order on disk
            train_indices = sorted(perm[:n_train])
            eval_indices  = sorted(perm[n_train : n_train + n_eval])
            test_indices  = sorted(perm[n_train + n_eval :])

            for split_name, indices in zip(_SPLIT_DIRS, [train_indices, eval_indices, test_indices]):
                split_dir = ds_dir / split_name
                split_dir.mkdir()
                for i in indices:
                    shutil.move(str(frame_dirs[i]), str(split_dir / frame_dirs[i].name))

            print(f"  [split] {output_dir_name}: "
                  f"{n_train} train / {n_eval} eval / {n_test} test  (seed={seed})")

    print(f"\n{'─' * 60}")
    print("  Split done.")
    print(f"{'─' * 60}\n")


def cmd_verify(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)

    selected = DATASETS
    if args.dataset:
        selected = [d for d in DATASETS if args.dataset.lower() in d["name"].lower()]
        if not selected:
            print(f"No dataset matching '{args.dataset}'.")
            sys.exit(1)

    # Collect results: (dataset_name, split_name, total, complete, error_details)
    # error_details: [(frame_name, exr_name, issue)]
    results = []

    for dataset in selected:
        for output_dir_name in dataset.get("output_dirs", []):
            ds_dir = out_dir / output_dir_name
            if not ds_dir.exists():
                print(f"  [skip] {output_dir_name} — not found at {ds_dir}")
                continue

            splits = list(_SPLIT_DIRS) if (ds_dir / "train").exists() else ["(flat)"]

            for split_name in splits:
                split_dir = ds_dir if split_name == "(flat)" else ds_dir / split_name
                if not split_dir.exists():
                    continue

                frame_dirs = _frame_dirs(split_dir)
                if not frame_dirs:
                    continue

                frame_errors: dict = {}
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futures = {ex.submit(_verify_frame, fd): fd for fd in frame_dirs}
                    for fut in tqdm(
                        as_completed(futures), total=len(frame_dirs),
                        desc=f"{output_dir_name}/{split_name}", ncols=80,
                    ):
                        fd  = futures[fut]
                        errs = fut.result()
                        if errs:
                            frame_errors[fd.name] = errs

                error_details = [
                    (fname, exr, issue)
                    for fname, errs in sorted(frame_errors.items())
                    for exr, issue in errs
                ]
                results.append((
                    output_dir_name, split_name,
                    len(frame_dirs),
                    len(frame_dirs) - len(frame_errors),
                    error_details,
                ))

    # ── Summary table ─────────────────────────────────────────────────────────
    COL   = (22, 7, 8, 10, 9, 9)
    HDR   = ("Dataset", "Split", "Frames", "Complete", "Missing", "Corrupt")
    SEP   = "─" * (sum(COL) + len(COL) * 2 + 1)

    def _row(*vals):
        return " " + "  ".join(str(v).ljust(c) for v, c in zip(vals, COL))

    print(f"\n{SEP}")
    print(_row(*HDR))
    print(SEP)

    grand_frames = grand_complete = grand_missing = grand_corrupt = 0
    all_errors: list = []

    for name, split, total, complete, errors in results:
        missing = sum(1 for *_, issue in errors if issue == "missing")
        corrupt = sum(1 for *_, issue in errors if issue == "corrupt")
        print(_row(name, split, total, complete, missing, corrupt))
        grand_frames   += total
        grand_complete += complete
        grand_missing  += missing
        grand_corrupt  += corrupt
        all_errors.extend((name, split, f, e, i) for f, e, i in errors)

    print(SEP)
    print(_row("TOTAL", "", grand_frames, grand_complete, grand_missing, grand_corrupt))
    print(f"{SEP}\n")

    if all_errors:
        print(f"{len(all_errors)} issue(s) found:\n")
        for ds_name, split, frame, exr, issue in all_errors:
            print(f"  {ds_name} / {split} / {frame} / {exr}  —  {issue}")
        print()
    else:
        print("  All files verified OK.\n")


def cmd_visualize(args: argparse.Namespace) -> None:
    visualize_frame(args.dataset, args.frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="setup_datasets",
        description="UnrealMVS dataset pipeline — download, convert, visualize",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Download, extract and convert datasets")
    p_setup.add_argument("--raw",       default="raw_datasets")
    p_setup.add_argument("--out",       default="datasets")
    p_setup.add_argument("--dataset",   default=None, help="Substring match on dataset name")
    p_setup.add_argument("--panorama",  action="store_true",
                         help="Also generate Equirectangular.exr panoramas (skips existing)")
    p_setup.add_argument("--split",     action="store_true",
                         help="After converting, assign frames to train/eval/test subdirs")
    p_setup.add_argument("--seed",      type=int, default=42,
                         help="RNG seed for --split (default: 42)")
    p_setup.add_argument("--verify",    action="store_true",
                         help="After splitting, verify all EXR files and print a summary")
    p_setup.add_argument("--visualize", action="store_true",
                         help="Also generate overview PNGs (skips existing)")
    p_setup.add_argument("--workers",   type=int, default=8)

    p_conv = sub.add_parser("convert", help="Convert already-downloaded raw data")
    p_conv.add_argument("--raw",           default="raw_datasets")
    p_conv.add_argument("--out",           default="datasets")
    p_conv.add_argument("--dataset",       default=None)
    p_conv.add_argument("--panorama",      action="store_true",
                        help="Also generate panoramas (skips existing)")
    p_conv.add_argument("--panorama-only", action="store_true",
                        help="Only (re)generate panoramas, skip image conversion")
    p_conv.add_argument("--workers",       type=int, default=8)

    p_split = sub.add_parser("split", help="Assign converted frames to train/eval/test subdirs")
    p_split.add_argument("--out",     default="datasets")
    p_split.add_argument("--dataset", default=None, help="Substring match on dataset name")
    p_split.add_argument("--seed",    type=int, default=42,
                         help="RNG seed for deterministic frame assignment (default: 42)")

    p_ver = sub.add_parser("verify", help="Check all EXR files and print a per-split summary")
    p_ver.add_argument("--out",     default="datasets")
    p_ver.add_argument("--dataset", default=None, help="Substring match on dataset name")
    p_ver.add_argument("--workers", type=int, default=8)

    p_vis = sub.add_parser("visualize", help="Generate coverage overview PNG for one frame")
    p_vis.add_argument("--dataset", default="datasets/Urban_Sunny")
    p_vis.add_argument("--frame",   default="0001")

    args = parser.parse_args()
    {
        "setup":     cmd_setup,
        "convert":   cmd_convert,
        "split":     cmd_split,
        "verify":    cmd_verify,
        "visualize": cmd_visualize,
    }[args.command](args)


if __name__ == "__main__":
    main()
