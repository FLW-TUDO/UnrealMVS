import cv2


def load_mask(mask_path: str) -> 'np.ndarray':
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: '{mask_path}'")
    return mask.astype(bool)
