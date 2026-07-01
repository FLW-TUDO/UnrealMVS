import os
import cv2


class CameraModel:
    def __init__(self, name, camera_type, intrinsic):
        self.invalid_mask = None
        self.name = name
        self.camera_type = camera_type
        self.cx = float(intrinsic['cx'])
        self.cy = float(intrinsic['cy'])
        self.resolution = intrinsic['resolution']  # [H, W]

        self.cam2rig = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rig2cam = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def set_mask(self, mask_path, mask_file):
        path = os.path.join(mask_path, mask_file)
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.invalid_mask = mask > 0

    def set_extrinsic(self, extrinsic):
        raise NotImplementedError

    def pixel_to_ray(self, p, out_theta=False, max_theta=None):
        raise NotImplementedError

    def ray_to_pixel(self, P, out_theta=False, max_theta=None):
        raise NotImplementedError
