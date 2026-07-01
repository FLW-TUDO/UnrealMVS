import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.pipelines.pipeline import Pipeline
from src.training.pipelines.amp import AMP
from src.log.tensorboard_logger import TensorBoard_Logger


def _make_dataloader(dataset, config, shuffle=None):
    dl_cfg     = config.parameters.training.dataloader
    batch_size = int(config.parameters.training.batch_size or 1)
    use_shuffle = dl_cfg.shuffle if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = use_shuffle,
        num_workers        = dl_cfg.num_workers,
        pin_memory         = dl_cfg.pin_memory,
        persistent_workers = dl_cfg.persistent_workers,
        prefetch_factor    = dl_cfg.prefetch_factor,
    )


class Trainer:
    def __init__(self, config, logger, model, train_dataset, val_dataset):
        self.config = config
        general    = config.parameters.general
        training   = config.parameters.training
        validation = config.parameters.validation
        logging    = config.parameters.logging
        ds_cfg     = config.datasets[0]

        self.device_str = str(general.device) if torch.cuda.is_available() else 'cpu'
        self.device     = torch.device(self.device_str)
        self.model      = model.to(self.device)

        weights_path = str(config.parameters.model.weights_path or "")
        if weights_path:
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint)
                logger.log(f"Resumed from weights: {weights_path}")
            except Exception as e:
                logger.log(f"Failed to load weights '{weights_path}': {e}", "error")

        self.epochs              = ds_cfg.epochs
        self.validation_interval = validation.interval
        self.use_tensorboard     = logging.use_tensorboard
        self.disable_tqdm        = logging.disable_tqdm
        self.name                = general.name

        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr       = ds_cfg.optimizer.lr,
            momentum = ds_cfg.optimizer.momentum,
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones = ds_cfg.optimizer.scheduler.steps,
            gamma      = ds_cfg.optimizer.scheduler.gamma,
        )

        # TensorBoard
        self.log_dir = os.path.join(str(logging.log_dir), str(general.name))
        if self.use_tensorboard:
            self.timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
            self.tb              = TensorBoard_Logger(self.tensorboard_dir, self.timestamp)
        else:
            self.tb = None

        # Datasets / DataLoaders
        self.train_dataset  = train_dataset
        self.val_dataset    = val_dataset
        self.train_loader = _make_dataloader(train_dataset, config)
        self.val_loader   = _make_dataloader(val_dataset, config, shuffle=False)


        self.current_epoch        = 0

        # Pipeline
        if training.pipeline == "amp":
            self.pipeline = AMP(self.device, config, logger, self.tb)
        else:
            self.pipeline = Pipeline(self.device, config, logger, self.tb)
        self.logger = logger

        logger.log(
            f"──────────────────────────────────────────\n"
            f"  Dataset : {train_dataset.name}  "
            f"(train: {len(train_dataset)}, val: {len(val_dataset)})\n"
            f"  Device  : {self.device_str}\n"
            f"  Pipeline: {'AMP (float16 autocast)' if training.pipeline == 'amp' else 'default (float32)'}\n"
            f"  Loss    : {training.loss_type}\n"
            f"  Epochs  : {self.epochs}  |  "
            f"LR: {ds_cfg.optimizer.lr}  →  "
            f"×{ds_cfg.optimizer.scheduler.gamma} at epochs {list(ds_cfg.optimizer.scheduler.steps)}\n"
            f"──────────────────────────────────────────"
        )
        time.sleep(0.05)

    # ── Finetuning on a new dataset ─────────────────────────────────────────────

    def update(self, train_dataset, val_dataset, config, idx):
        ds_cfg             = config.datasets[idx]
        self.name          = f"{config.parameters.general.name}_{train_dataset.name}"
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.pipeline.name = self.name
        self.train_loader  = _make_dataloader(train_dataset, config)
        dl_cfg = config.parameters.training.dataloader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size         = 1,
            shuffle            = False,
            num_workers        = dl_cfg.num_workers,
            pin_memory         = dl_cfg.pin_memory,
            persistent_workers = dl_cfg.persistent_workers,
            prefetch_factor    = dl_cfg.prefetch_factor,
        )
        self.epochs = ds_cfg.epochs

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr       = ds_cfg.optimizer.lr,
            momentum = ds_cfg.optimizer.momentum,
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones = ds_cfg.optimizer.scheduler.steps,
            gamma      = ds_cfg.optimizer.scheduler.gamma,
        )

        training = config.parameters.training
        self.logger.log(
            f"──────────────────────────────────────────\n"
            f"  Dataset : {train_dataset.name}  "
            f"(train: {len(train_dataset)}, val: {len(val_dataset)})\n"
            f"  Device  : {self.device_str}\n"
            f"  Pipeline: {'AMP (float16 autocast)' if training.pipeline == 'amp' else 'default (float32)'}\n"
            f"  Loss    : {training.loss_type}\n"
            f"  Epochs  : {self.epochs}  |  "
            f"LR: {ds_cfg.optimizer.lr}  →  "
            f"×{ds_cfg.optimizer.scheduler.gamma} at epochs {list(ds_cfg.optimizer.scheduler.steps)}\n"
            f"──────────────────────────────────────────"
        )
        time.sleep(0.05)

    # ── Main training loop ──────────────────────────────────────────────────────

    def train(self):
        if self.epochs == 0:
            self.logger.log(f"Skipping training on '{self.train_dataset.name}' (epochs=0)")
            return
        self.logger.log(f"Starting training on '{self.train_dataset.name}' ...")
        for epoch in range(self.epochs):
            self.model.train()
            train_stats = self._train_one_epoch(self.current_epoch)

            if self.use_tensorboard:
                self.tb.log_learning_rate(self.name, self.scheduler.get_last_lr()[0], self.current_epoch)

            val_stats = self._validate_one_epoch(self.current_epoch)

            self.logger.log(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.log(
                f"  Train  — loss: {train_stats['loss']:.4f}  MAE: {train_stats['mae']:.4f}  "
                f"RMSE: {train_stats['rmse']:.4f}  E1: {train_stats['e1']:.2f}%  "
                f"E3: {train_stats['e3']:.2f}%  E5: {train_stats['e5']:.2f}%"
            )
            self.logger.log(
                f"  Val    — loss: {val_stats['loss']:.4f}  MAE: {val_stats['mae']:.4f}  "
                f"RMSE: {val_stats['rmse']:.4f}  E1: {val_stats['e1']:.2f}%  "
                f"E3: {val_stats['e3']:.2f}%  E5: {val_stats['e5']:.2f}%"
            )

            ckpt_base = f"{self.tensorboard_dir}/{self.timestamp}" if self.use_tensorboard else self.log_dir
            self.pipeline.save_checkpoint(self.model, f"{ckpt_base}/latest", self.name)
            self.pipeline.save_checkpoint(self.model, self.log_dir,           f"{self.name}_latest")

            self.scheduler.step()
            self.current_epoch += 1

        if self.tb:
            self.tb.close()

    def _train_one_epoch(self, epoch):
        running = {k: 0.0 for k in ('loss', 'error', 'e1', 'e3', 'e5', 'mae', 'rmse')}
        batch_count = 0
        last_data   = None
        loader = iter(tqdm(self.train_loader, desc='Training', leave=False, ncols=100, disable=self.disable_tqdm))
        while True:
            try:
                data = next(loader)
            except StopIteration:
                break
            except Exception as e:
                self.logger.log(f"Skipping bad training sample: {e}")
                continue
            try:
                stats = self.pipeline.train_step(self.model, data, self.optimizer)
                if np.isnan(stats['loss']):
                    continue
                last_data = data
            except Exception as e:
                self.logger.log(str(e))
                continue
            for k in running:
                running[k] += stats[k]
            batch_count += 1

        if last_data is not None and self.use_tensorboard:
            self.pipeline.log_sample_image(self.model, last_data, epoch)

        if batch_count == 0:
            return {k: 0.0 for k in running}
        totals = {k: v / batch_count for k, v in running.items()}
        if self.use_tensorboard:
            self.tb.log_metrics(self.name, {'loss': totals['loss'], 'error': totals['error']}, epoch, 'Training')
            self.tb.log_detailed_errors(self.name, epoch, {
                'error_1%': totals['e1'], 'error_3%': totals['e3'],
                'error_5%': totals['e5'], 'mae': totals['mae'], 'rmse': totals['rmse'],
            }, 'Training')
        return totals

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        running = {k: 0.0 for k in ('loss', 'error', 'e1', 'e3', 'e5', 'mae', 'rmse')}
        batch_count = 0
        save_image  = True

        with torch.no_grad():
            loader = iter(tqdm(self.val_loader, desc='Validation', leave=False, disable=self.disable_tqdm))
            while True:
                try:
                    data = next(loader)
                except StopIteration:
                    break
                except Exception as e:
                    self.logger.log(f"Skipping bad validation sample: {e}")
                    continue
                try:
                    stats = self.pipeline.validation_step(self.model, data, epoch, save_image)
                    if np.isnan(stats['loss']):
                        continue
                    save_image = False
                except Exception as e:
                    self.logger.log(str(e))
                    continue
                for k in running:
                    running[k] += stats[k]
                batch_count += 1

        if batch_count == 0:
            return {k: 0.0 for k in running}
        totals = {k: v / batch_count for k, v in running.items()}
        if self.use_tensorboard:
            self.tb.log_metrics(self.name, {'loss': totals['loss']}, epoch, 'Validation')
            self.tb.log_detailed_errors(self.name, epoch, {
                'error_1%': totals['e1'], 'error_3%': totals['e3'],
                'error_5%': totals['e5'], 'mae': totals['mae'], 'rmse': totals['rmse'],
            }, 'Validation')
        return totals
