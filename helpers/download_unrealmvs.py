"""
Download UnrealMVS datasets from HuggingFace.

Usage:
  python helpers/download_unrealmvs.py                                    # all scenes
  python helpers/download_unrealmvs.py --scene factory                    # one scene (fuzzy)
  python helpers/download_unrealmvs.py --scene factory --scene tokyo      # multiple scenes
  python helpers/download_unrealmvs.py --scene polar --split              # download + split
  python helpers/download_unrealmvs.py --list                             # list available scenes
  python helpers/download_unrealmvs.py --out /data/datasets               # custom output dir

Split sizes (per scene, configurable via --train-ratio/--eval-ratio):
  Default: 70% train, 20% eval, 10% test
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

HF_REPO = "flw-tu-dortmund/UnrealMVS"

SCENES = [
    "Bazaar",
    "Cambodia",
    "Catacomb",
    "Cigar Room",
    "CitySampleDay",
    "CitySampleNight",
    "ElectricDreamsEnv",
    "Engine Hall",
    "FactoryEnvironment",
    "Mediterranean",
    "PolarStation",
    "PoliceHeadquarters",
    "Tokyo",
    "WindyHouse",
]


def _resolve_scenes(queries: list[str]) -> list[str]:
    result, seen = [], set()
    for query in queries:
        matches = [s for s in SCENES if query.lower() in s.lower()]
        if not matches:
            print(f"No scene matching '{query}'. Run --list to see available scenes.")
            sys.exit(1)
        for m in matches:
            if m not in seen:
                result.append(m)
                seen.add(m)
    return result


def cmd_list() -> None:
    print(f"\nAvailable scenes in {HF_REPO}:\n")
    for s in SCENES:
        print(f"  {s}")
    print()


def _frame_dirs(scene_dir: Path) -> list[Path]:
    return sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit())


def split_scene(scene_dir: Path, train_ratio: float, eval_ratio: float, seed: int) -> None:
    if (scene_dir / "train").exists():
        print(f"  [skip split]  '{scene_dir.name}' already split")
        return

    frames  = _frame_dirs(scene_dir)
    n       = len(frames)
    n_train = round(n * train_ratio)
    n_eval  = round(n * eval_ratio)
    n_test  = n - n_train - n_eval

    if n < 3:
        print(f"  [error split] '{scene_dir.name}' has only {n} frames — skipping")
        return

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)
    splits = {
        "train": sorted(perm[:n_train]),
        "eval":  sorted(perm[n_train : n_train + n_eval]),
        "test":  sorted(perm[n_train + n_eval :]),
    }

    for split_name, indices in splits.items():
        split_dir = scene_dir / split_name
        split_dir.mkdir()
        for i in indices:
            shutil.move(str(frames[i]), str(split_dir / frames[i].name))

    print(f"  [split]  '{scene_dir.name}': "
          f"{n_train} train / {n_eval} eval / {n_test} test  (seed={seed})")


def cmd_download(scenes: list[str], out_dir: Path, do_split: bool = False,
                 train_ratio: float = 0.7, eval_ratio: float = 0.2,
                 seed: int = 42) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed.")
        print("        Run: pip install huggingface_hub")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir.resolve()}")

    for scene in scenes:
        dest = out_dir / scene
        if dest.exists() and any(dest.iterdir()):
            print(f"\n[skip]  '{scene}'  (already downloaded)")
        else:
            print(f"\n{'─' * 60}")
            print(f"  Downloading: {scene}")
            print(f"{'─' * 60}")

            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=out_dir,
                allow_patterns=[f"{scene}/**"],
            )
            print(f"  -> {dest}")

        if do_split:
            split_scene(dest, train_ratio, eval_ratio, seed)

    print(f"\n{'─' * 60}")
    print("  Download complete.")
    print(f"{'─' * 60}")

    if not do_split:
        print()
        print("  Next step — assign train/eval/test splits:")
        print()
        print("    python helpers/download_unrealmvs.py --split --scene <name>")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download UnrealMVS datasets from HuggingFace."
    )
    parser.add_argument(
        "--scene", metavar="NAME", action="append", dest="scenes",
        help="Scene to download (case-insensitive substring match). "
             "Repeat to select multiple: --scene factory --scene tokyo. "
             "Omit to download all scenes.",
    )
    parser.add_argument(
        "--out", default="./datasets", metavar="DIR",
        help="Output directory (default: ./datasets).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenes and exit.",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Assign train/eval/test splits after downloading.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, metavar="R",
        help="Fraction of frames for training (default: 0.7).",
    )
    parser.add_argument(
        "--eval-ratio", type=float, default=0.2, metavar="R",
        help="Fraction of frames for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible splits (default: 42).",
    )
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    scenes = _resolve_scenes(args.scenes) if args.scenes else SCENES
    cmd_download(
        scenes, Path(args.out),
        do_split=args.split,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
