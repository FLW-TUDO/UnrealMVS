import sys
import numpy as np

from src.camera.model.camera import CameraModel
import src.utils.array as array
import src.utils.geometry as geometry

EPS = sys.float_info.epsilon


class OcamModel(CameraModel):
    def __init__(self, name, camera_type, intrinsic):
        super().__init__(name, camera_type, intrinsic)

        self.xc       = float(intrinsic['cx'])
        self.yc       = float(intrinsic['cy'])
        self.c        = float(intrinsic['affine'][0])
        self.d        = float(intrinsic['affine'][1])
        self.e        = float(intrinsic['affine'][2])
        self.poly     = intrinsic['poly'][-1:0:-1]
        self.inv_poly = intrinsic['inv_poly'][-1:0:-1]
        self.max_theta = np.deg2rad(intrinsic['max_fov']) / 2

        self.cam2rig = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rig2cam = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def set_extrinsic(self, extrinsic):
        orientation = extrinsic['rig2cam']['orientation']
        position    = extrinsic['rig2cam']['translation']
        self.rig2cam = np.array(orientation + position).reshape(6, 1)
        self.cam2rig = geometry.inverseTransform(self.rig2cam)

    def pixel_to_ray(self, p, out_theta=False, max_theta=None):
        if max_theta is None:
            max_theta = self.max_theta
        x = p[1, :].reshape((1, -1)) - self.xc
        y = p[0, :].reshape((1, -1)) - self.yc
        p = array.concat((x, y), axis=0)
        invdet = 1.0 / (self.c - self.d * self.e)
        A_inv  = invdet * np.array([[1, -self.d], [-self.e, self.c]])
        p = A_inv.dot(p)
        x = p[1, :].reshape((1, -1))
        y = p[0, :].reshape((1, -1))
        rho   = array.sqrt(x * x + y * y)
        z     = array.polyval(self.poly, rho).reshape((1, -1))
        theta = array.atan2(rho, -z)
        out   = array.concat((x, y, -z), axis=0)
        out[:, theta.squeeze() > max_theta] = np.nan
        if out_theta:
            return out, theta
        return out

    def ray_to_pixel(self, P, out_theta=False, max_theta=None):
        if max_theta is None:
            max_theta = self.max_theta
        norm  = array.sqrt(P[0, :] ** 2 + P[1, :] ** 2) + EPS
        theta = array.atan2(-P[2, :], norm)
        rho   = array.polyval(self.inv_poly, theta)
        theta = theta + np.pi / 2
        x  = P[1, :] / norm * rho
        y  = P[0, :] / norm * rho
        x2 = x * self.c + y * self.d + self.xc
        y2 = x * self.e + y + self.yc
        x2 = x2.reshape((1, -1))
        y2 = y2.reshape((1, -1))
        out = array.concat((y2, x2), axis=0)
        out[:, theta.squeeze() > max_theta] = -1.0
        if out_theta:
            return out, theta
        return out
