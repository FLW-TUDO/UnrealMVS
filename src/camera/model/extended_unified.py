import numpy as np
from scipy.spatial.transform import Rotation as R

from src.camera.model.camera import CameraModel


class EUCMModel(CameraModel):
    def __init__(self, name, camera_type, intrinsic):
        super().__init__(name, camera_type, intrinsic)
        self.fx    = float(intrinsic['fx'])
        self.fy    = float(intrinsic['fy'])
        self.alpha = float(intrinsic['alpha'])
        self.beta  = float(intrinsic['beta'])

        self.cam2rig = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rig2cam = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def set_extrinsic(self, extrinsic):
        if 'rig2cam' not in extrinsic:
            raise ValueError("EUCMModel expects rig2cam in extrinsics")
        quaternion  = extrinsic['rig2cam']['quaternion']
        translation = extrinsic['rig2cam']['translation']
        rot   = R.from_quat(quaternion).as_matrix()
        trans = np.array(translation).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3:]  = trans
        T_inv     = np.linalg.inv(T)
        rot_inv   = T_inv[:3, :3]
        trans_inv = T_inv[:3, 3]
        quat_inv  = R.from_matrix(rot_inv).as_quat()
        self.rig2cam = np.array(list(quat_inv) + list(trans_inv)).reshape(7, 1)

    def ray_to_pixel(self, p_3d, out_theta=False, max_theta=None):
        x, y, z = p_3d[0], p_3d[1], p_3d[2]
        eps = 1e-9

        r2     = x * x + y * y
        r_sqrt = np.sqrt(r2)

        u = np.full_like(x, -1e5)
        v = np.full_like(y, -1e5)

        rho2      = self.beta * r2 + z * z
        rho       = np.sqrt(rho2)
        w         = self.alpha / (1 - self.alpha) if self.alpha > 0.5 else (1 - self.alpha) / self.alpha
        valid_z   = z > -w * rho
        mask      = (r_sqrt > eps) & valid_z
        norm      = self.alpha * rho + (1 - self.alpha) * z

        u[mask] = self.fx * x[mask] / norm[mask] + self.cx
        v[mask] = self.fy * y[mask] / norm[mask] + self.cy

        u   = u.reshape((1, -1))
        v   = v.reshape((1, -1))
        out = np.concatenate((u, v), axis=0)

        theta = np.zeros_like(r_sqrt)
        theta[mask] = np.arctan2(r_sqrt[mask], z[mask])

        if out_theta:
            return out, theta
        return out
