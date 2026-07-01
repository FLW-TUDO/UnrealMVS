import numpy as np
from scipy.spatial.transform import Rotation as R

from src.camera.model.camera import CameraModel


class KBModel(CameraModel):
    def __init__(self, name, camera_type, intrinsic):
        super().__init__(name, camera_type, intrinsic)
        self.fx = float(intrinsic['fx'])
        self.fy = float(intrinsic['fy'])
        self.k0 = float(intrinsic['k0'])
        self.k1 = float(intrinsic['k1'])
        self.k2 = float(intrinsic['k2'])
        self.k3 = float(intrinsic['k3'])

    def set_extrinsic(self, extrinsic):
        if 'rig2cam' not in extrinsic:
            raise ValueError("KBModel expects rig2cam in extrinsics")
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

        theta            = np.arctan2(r_sqrt, z)
        use_dist_mask    = r_sqrt > eps
        fallback_mask    = ~use_dist_mask

        if np.any(use_dist_mask):
            t  = theta[use_dist_mask]
            t2 = t * t
            r_theta  = self.k3 * t2
            r_theta += self.k2
            r_theta *= t2
            r_theta += self.k1
            r_theta *= t2
            r_theta += self.k0
            r_theta *= t2
            r_theta += 1.0
            r_theta *= t
            u[use_dist_mask] = self.fx * x[use_dist_mask] * r_theta / r_sqrt[use_dist_mask] + self.cx
            v[use_dist_mask] = self.fy * y[use_dist_mask] * r_theta / r_sqrt[use_dist_mask] + self.cy

        if np.any(fallback_mask):
            valid_z = z[fallback_mask] >= eps
            idx     = np.where(fallback_mask)[0][valid_z]
            u[idx]  = self.fx * x[idx] / z[idx] + self.cx
            v[idx]  = self.fy * y[idx] / z[idx] + self.cy

        u   = u.reshape((1, -1))
        v   = v.reshape((1, -1))
        out = np.concatenate((u, v), axis=0)
        if out_theta:
            return out, theta
        return out
