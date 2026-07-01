import os
import sys

import torch

from src.training.error import eval_error
from src.training.loss_function import (
    absolute_error_loss, smooth_l1_loss, entropy_boundary_loss,
)
from src.utils.depth import index_to_invdepth

EPS = sys.float_info.epsilon


class Pipeline:
    def __init__(self, device, config, logger, tensorboard_logger):
        self.device    = device
        self.name      = config.parameters.general.name
        self.logger    = logger
        self.tb        = tensorboard_logger
        self.loss_type = config.parameters.training.loss_type
        self.weight_l  = 1.0

        sweep = config.parameters.spherical_sweep
        self.num_invdepth         = sweep.num_invdepth
        self.min_invdepth         = 1.0 / sweep.max_depth
        self.sample_step_invdepth = (
            (1.0 / sweep.min_depth - self.min_invdepth) / (sweep.num_invdepth - 1.0)
        )

    # ── Data preparation ────────────────────────────────────────────────────────

    def _prepare_data(self, data):
        images, gts, valid, _, grids = data
        images = [torch.Tensor(img).to(self.device) for img in images]
        gts    = torch.Tensor(gts).to(self.device)
        valid  = torch.as_tensor(valid).to(self.device)
        grids  = [torch.as_tensor(grid[0]).to(self.device) for grid in grids]
        return images, gts, valid, grids

    # ── Loss ────────────────────────────────────────────────────────────────────

    def calculate_loss(self, invdepth_idx, prob, gts, valid):
        if self.loss_type == "mean":
            return absolute_error_loss(invdepth_idx, gts, valid)
        if self.loss_type == "smooth_l1":
            return smooth_l1_loss(invdepth_idx, gts, valid)
        if self.loss_type == "huber":
            entropy = torch.sum(-torch.log(prob + EPS) * prob, 0)
            return (smooth_l1_loss(invdepth_idx, gts, valid)
                    + self.weight_l * entropy_boundary_loss(entropy, 6, 1))
        raise ValueError(f"Unknown loss_type: '{self.loss_type}'")

    # ── Validation result processing ────────────────────────────────────────────

    def _process_validation_results(self, invdepth_idx, gts, valid, epoch, save_image):
        error = torch.abs(invdepth_idx[valid] - gts[valid])
        loss  = torch.nanmean(error)
        e1, e3, e5, mae, rmse = eval_error(invdepth_idx, gts, valid, self.num_invdepth)

        if save_image:
            self.tb.log_prediction(
                dataset_name=self.name,
                epoch=epoch,
                predictions={
                    'ground_truth': index_to_invdepth(
                        gts.float().squeeze(0), self.min_invdepth, self.sample_step_invdepth
                    ),
                    'prediction': index_to_invdepth(
                        invdepth_idx.float().squeeze(0), self.min_invdepth, self.sample_step_invdepth
                    ),
                },
            )
        return {
            'loss': loss.item(), 'error': error.nanmean().item(),
            'e1': e1, 'e3': e3, 'e5': e5, 'mae': mae, 'rmse': rmse,
        }

    # ── Checkpoint ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, model, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{model_name}.pt')
        torch.save(model.state_dict(), path)

    # ── Train / validation steps ────────────────────────────────────────────────

    def train_step(self, model, data, optimizer):
        images, gts, valid, grids = self._prepare_data(data)
        optimizer.zero_grad(set_to_none=True)
        invdepth_idx, prob, _ = model(images, grids, out_cost=True)
        error = torch.abs(invdepth_idx[valid] - gts[valid])
        loss  = self.calculate_loss(invdepth_idx, prob, gts, valid)
        loss.backward()
        optimizer.step()
        e1, e3, e5, mae, rmse = eval_error(invdepth_idx, gts, valid, self.num_invdepth)
        return {
            'loss': loss.item(), 'error': error.nanmean().item(),
            'e1': e1, 'e3': e3, 'e5': e5, 'mae': mae, 'rmse': rmse,
        }

    def log_sample_image(self, model, data, epoch):
        if self.tb is None:
            return
        images, gts, _, grids = self._prepare_data(data)
        with torch.no_grad():
            invdepth_idx = model(images, grids)
        self.tb.log_prediction(
            dataset_name=self.name,
            epoch=epoch,
            predictions={
                'ground_truth': index_to_invdepth(
                    gts.float().squeeze(0), self.min_invdepth, self.sample_step_invdepth
                ),
                'prediction': index_to_invdepth(
                    invdepth_idx.float().squeeze(0), self.min_invdepth, self.sample_step_invdepth
                ),
            },
        )

    def validation_step(self, model, data, epoch, save_image=False):
        images, gts, valid, grids = self._prepare_data(data)
        invdepth_idx = model(images, grids)
        return self._process_validation_results(invdepth_idx, gts, valid, epoch, save_image)
