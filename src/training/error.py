import numpy as np
import src.utils.array as array


def eval_error(invdepth_idx, gt, valid, num_invdepth) -> tuple:
    """Compute E1/E3/E5/MAE/RMSE between predicted and GT inverse-depth indices."""
    pred  = array.toNumpy(invdepth_idx).flatten()
    gt    = array.toNumpy(gt).flatten()
    valid = array.toNumpy(valid).flatten().astype(bool)
    valid = valid & ~np.isnan(pred)

    n_valid = float(np.sum(valid))
    if n_valid == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    error = np.abs(pred[valid] - gt[valid]) / num_invdepth * 100
    e1    = np.sum(error > 1) / n_valid * 100
    e3    = np.sum(error > 3) / n_valid * 100
    e5    = np.sum(error > 5) / n_valid * 100
    mae   = np.nanmean(error)
    rmse  = np.sqrt(np.nanmean(error ** 2))
    return e1, e3, e5, mae, rmse
