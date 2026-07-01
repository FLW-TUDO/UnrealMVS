import numpy as np

from src.camera.model.camera import CameraModel


class EquidistantModel(CameraModel):
    def __init__(self, name, camera_type, intrinsic):
        super().__init__(name, camera_type, intrinsic)
        self.fov = float(intrinsic['fov'])

        self.cam2rig = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rig2cam = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def set_extrinsic(self, extrinsic):
        if 'rig2cam' not in extrinsic:
            raise ValueError("EquidistantModel expects rig2cam in extrinsics")
        inv_quaternion  = extrinsic['rig2cam']['quaternion']
        inv_translation = extrinsic['rig2cam']['translation']
        quaternion      = extrinsic['cam2rig']['quaternion']
        translation     = extrinsic['cam2rig']['translation']
        self.cam2rig = np.array(quaternion + translation).reshape(7, 1)
        self.rig2cam = np.array(inv_quaternion + inv_translation).reshape(7, 1)

    def ray_to_pixel(self, p, out_theta=False, max_theta=None):
        x, y, z   = p[0, :], p[1, :], p[2, :]
        r         = np.sqrt(x ** 2 + y ** 2)
        theta     = np.arctan2(r, z)
        fov_rad   = self.fov / 180 * np.pi
        half_fov  = fov_rad / 2.0
        r_theta   = 2.0 * theta / fov_rad

        r_safe = np.where(r > 1e-8, r, 1e-8)
        u = (self.cx + r_theta * x / r_safe * self.cx).reshape((1, -1))
        v = (self.cy + r_theta * y / r_safe * self.cy).reshape((1, -1))

        # Strahlen außerhalb des FOV auf -1 setzen (wie OcamModel)
        outside = (theta > half_fov).reshape(1, -1)
        u = np.where(outside, -1.0, u)
        v = np.where(outside, -1.0, v)

        out = np.concatenate((u, v), axis=0)
        if out_theta:
            return out, theta
        return out
