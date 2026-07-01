import os
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoard_Logger:
    def __init__(self, base_dir: str, timestamp: str):
        self.base_dir  = base_dir
        self.timestamp = timestamp
        self.writers: Dict[str, SummaryWriter] = {}

    def _writer(self, dataset_name: str) -> SummaryWriter:
        if dataset_name not in self.writers:
            log_dir = os.path.join(self.base_dir, self.timestamp, dataset_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writers[dataset_name] = SummaryWriter(log_dir)
        return self.writers[dataset_name]

    def log_metrics(self, dataset_name: str, metrics: Dict[str, float], epoch: int, phase: str = "train"):
        w = self._writer(dataset_name)
        for name, value in metrics.items():
            w.add_scalar(f'{phase}/{name}', value, epoch)

    def log_detailed_errors(self, dataset_name: str, epoch: int,
                            error_metrics: Dict[str, float], phase: str = "train"):
        w = self._writer(dataset_name)
        for name, value in error_metrics.items():
            w.add_scalar(f'{phase}/errors/{name}', value, epoch)

    def log_prediction(self, dataset_name: str, epoch: int, predictions: Dict[str, torch.Tensor]):
        w = self._writer(dataset_name)
        for name, image in predictions.items():
            w.add_image(f'predictions/{name}', image, epoch, dataformats='HW')

    def log_learning_rate(self, dataset_name: str, lr: float, epoch: int):
        self._writer(dataset_name).add_scalar('hyperparameters/learning_rate', lr, epoch)

    def close(self):
        for w in self.writers.values():
            w.close()
