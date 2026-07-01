import numpy as np
import torch
from tqdm import tqdm

from src.training.error import eval_error
from src.utils.array import cast_precision
import src.utils.array as array


class Evaluator:
    def __init__(self, model, device, config, logger):
        self.model   = model
        self.device  = device
        self.config  = config
        self.logger  = logger
        self.dataset = None
        self.dataset_name = "Unknown"

    def set_dataset(self, dataset):
        self.dataset      = dataset
        self.dataset_name = f"{self.config.parameters.general.name}_{dataset.name}"

    def _prepare_model(self, precision: str):
        self.model.eval().to(self.device)
        if precision == "float16":
            self.model.half()
        elif precision == "bfloat16":
            self.model.bfloat16()
        else:
            self.model.float()

    def _infer(self, images, grids, precision: str):
        images = [torch.tensor(img).unsqueeze(0).to(self.device) for img in images]
        grids  = [torch.tensor(grid).to(self.device) for grid in grids]
        images = cast_precision(images, precision)
        grids  = cast_precision(grids, precision)
        with torch.no_grad():
            return self.model(torch.stack(images), torch.stack(grids))

    # ── Test with ground truth ──────────────────────────────────────────────────

    def test_model(self, precision="float32", iterations=None):
        self.logger.log(f"Evaluating model on {self.dataset_name} ...")
        self._prepare_model(precision)
        num_invdepth = self.config.parameters.spherical_sweep.num_invdepth

        n = min(iterations, self.dataset.data_size) if iterations else self.dataset.data_size
        e1 = e3 = e5 = mae = rmse = valid_ratio = 0.0

        for i in tqdm(range(n), disable=self.config.parameters.logging.disable_tqdm):
            images, gt_idx, valid, _, grids = self.dataset[i]
            pred_idx = self._infer(images, grids, precision)
            _e1, _e3, _e5, _mae, _rms = eval_error(pred_idx, gt_idx, valid, num_invdepth)
            e1 += _e1; e3 += _e3; e5 += _e5; mae += _mae; rmse += _rms
            valid_ratio += np.sum(valid) / valid.size

        self.logger.log(
            f"E1: {e1/n:.2f}%  E3: {e3/n:.2f}%  E5: {e5/n:.2f}%  "
            f"MAE: {mae/n:.4f}  RMSE: {rmse/n:.4f}  Valid: {valid_ratio/n:.4f}"
        )
