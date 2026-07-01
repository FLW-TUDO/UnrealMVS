import torch
import torch.nn.functional as F
import cv2, numpy as np

from src.training.error import eval_error
from src.training.pipelines.pipeline import Pipeline
from src.utils.depth import index_to_invdepth


class AMP(Pipeline):
    """Automatic Mixed Precision training pipeline."""

    def __init__(self, device, config, logger, tensorboard_logger):
        super().__init__(device, config, logger, tensorboard_logger)
        self.device_str = str(device) if torch.cuda.is_available() else 'cpu'
        self.scaler     = torch.amp.GradScaler()

    def _show_projected_images(self, images, grids):
        num_cams   = len(images)
        num_depths = grids[0].shape[0]
        d_indices  = [0, num_depths // 2, num_depths - 1]
        d_labels   = ['near', 'mid', 'far']

        rows = []
        for d_idx, d_label in zip(d_indices, d_labels):
            row_imgs = []
            for i in range(num_cams):
                proj = F.grid_sample(
                    images[i].float(), grids[i][d_idx].unsqueeze(0),
                    align_corners=True, padding_mode='zeros',
                )  # (1, 1, H_out, W_out)
                arr = (np.clip(proj.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(arr, (0, 0), (arr.shape[1] - 1, arr.shape[0] - 1), (0, 0, 255), 2)
                cv2.putText(arr, f'cam{i+1} {d_label}',
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                row_imgs.append(arr)
            rows.append(np.hstack(row_imgs))

        cv2.imshow("Projected (rows: near | mid | far)", np.vstack(rows))

    def train_step(self, model, data, optimizer):
        images, gts, valid, grids = self._prepare_data(data)


        """for i, img in enumerate(images):
            arr = (np.clip(img.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            cv2.imshow(f"FisheyeCamera_{i + 1} (raw)", arr)

        with torch.no_grad():
            self._show_projected_images(images, grids)

        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device_str):
            invdepth_idx, prob, _ = model(images, grids, out_cost=True)
            error = torch.abs(invdepth_idx[valid] - gts[valid])
            loss  = self.calculate_loss(invdepth_idx, prob, gts, valid)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        e1, e3, e5, mae, rmse = eval_error(invdepth_idx, gts, valid, self.num_invdepth)
        return {
            'loss': loss.item(), 'error': error.nanmean().item(),
            'e1': e1, 'e3': e3, 'e5': e5, 'mae': mae, 'rmse': rmse,
        }

    def log_sample_image(self, model, data, epoch):
        if self.tb is None:
            return
        images, gts, _, grids = self._prepare_data(data)
        with torch.no_grad(), torch.amp.autocast(self.device_str):
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
        with torch.amp.autocast(self.device_str):
            invdepth_idx = model(images, grids)
        return self._process_validation_results(invdepth_idx, gts, valid, epoch, save_image)
