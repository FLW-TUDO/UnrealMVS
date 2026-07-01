"""
UnrealMVS V2 — Training & Evaluation entry point.

Usage:
    python main.py --train    --config-file data/training.yaml
    python main.py --evaluate --config-file data/training.yaml --weights-path logs/best.pt
"""

import os
import sys
import time
import argparse

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import torch

from src.config import Config
from src.log.logger import Logger
from src.utils.visualize import draw_unrealmvs_ascii
from src.datasets.dataset import MultiDatasets
from src.datasets.validate import check_datasets
from src.models.neural_networks.omnimvs import OmniMVS
from src.training.trainer import Trainer


# ── Training ───────────────────────────────────────────────────────────────────

def train(config: Config, logger: Logger) -> None:
    model = OmniMVS(config)

    for idx, ds_cfg in enumerate(config.datasets):
        train_ds = MultiDatasets(ds_cfg, config, split="train")
        val_ds   = MultiDatasets(ds_cfg, config, split="eval")

        if idx == 0:
            logger.log(f"\n ---------- Training OmniMVS on {ds_cfg.name} ---------- \n")
            trainer = Trainer(config, logger, model, train_ds, val_ds)
        else:
            logger.log(f"\n ---------- Finetuning on {ds_cfg.name} ---------- \n")
            trainer.update(train_ds, val_ds, config, idx)

        trainer.train()
        trainer.pipeline.save_checkpoint(model, trainer.log_dir, f"{trainer.name}_latest")


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(config: Config, logger: Logger, args) -> None:
    from src.evaluation.run import run_evaluation

    weights = args.weights_path or str(config.parameters.model.weights_path or "")
    if not weights:
        logger.log("No weights path specified. Use --weights-path.", "error")
        return

    device = torch.device(
        str(config.parameters.general.device) if torch.cuda.is_available() else "cpu"
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        seen = []
        for ds_cfg in config.datasets:
            for name in ds_cfg.subsets:
                if name not in seen:
                    seen.append(name)
        datasets = seen

    weights_stem = os.path.splitext(os.path.basename(weights))[0]
    log_path = os.path.join(logger.log_dir, f"eval_{weights_stem}.txt")

    run_evaluation(
        config, weights, splits, datasets, args.precision, device,
        save_dir=args.save_dir, log_path=log_path,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UnrealMVS V2 — train or evaluate OmniMVS.")
    parser.add_argument("--train",        action="store_true", help="Run training.")
    parser.add_argument("--evaluate",     action="store_true", help="Run evaluation.")
    parser.add_argument("--config-file",  default="./data/training.yaml",
                        help="Path to YAML config (default: data/training.yaml).")
    parser.add_argument("--name",         help="Override config experiment name.")
    parser.add_argument("--weights-path", help="Path to .pt checkpoint for evaluation or training resume.")
    parser.add_argument("--min-depth",    type=float, help="Override min depth.")
    # evaluate-only options
    parser.add_argument("--splits",    default="test",
                        help="Comma-separated splits to evaluate (default: test).")
    parser.add_argument("--datasets",  default=None,
                        help="Comma-separated dataset names (default: all from config).")
    parser.add_argument("--precision", default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Inference precision (default: float32).")
    parser.add_argument("--save-dir",  default=None,
                        help="Save per-frame invdepth PNGs under this directory.")
    args = parser.parse_args()

    draw_unrealmvs_ascii()

    config = Config.from_yaml(args.config_file)
    if args.name:
        config.parameters.general.name = args.name
    if args.weights_path:
        config.parameters.model.weights_path = args.weights_path
    if args.min_depth:
        config.parameters.spherical_sweep.min_depth = args.min_depth

    log_dir  = str(config.parameters.logging.log_dir or "./logs")
    log_name = str(config.parameters.general.name    or "default")
    logger   = Logger(os.path.join(log_dir, log_name))
    logger.save_copy(args.config_file)

    t0 = time.time()
    logger.log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.train:
        if not check_datasets(config, splits=["train", "eval"]):
            sys.exit(1)
        train(config, logger)
    if args.evaluate:
        eval_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        if not check_datasets(config, splits=eval_splits):
            sys.exit(1)
        evaluate(config, logger, args)

    elapsed = time.time() - t0
    logger.log(f"Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
