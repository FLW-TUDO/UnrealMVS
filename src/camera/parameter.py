import json
import os
from pathlib import Path
from typing import List

from src.camera.model.camera import CameraModel
from src.camera.model.ocam import OcamModel
from src.camera.model.kannala_brandt import KBModel
from src.camera.model.extended_unified import EUCMModel
from src.camera.model.equidistant import EquidistantModel


def create_camera_model(camera_type: str, camera_name: str, intrinsic: dict) -> CameraModel:
    if camera_type == "ocam":
        return OcamModel(camera_name, camera_type, intrinsic)
    if camera_type == "kb":
        return KBModel(camera_name, camera_type, intrinsic)
    if camera_type == "eucm":
        return EUCMModel(camera_name, camera_type, intrinsic)
    if camera_type == "equidistant":
        return EquidistantModel(camera_name, camera_type, intrinsic)
    raise ValueError(f"Unsupported camera type: {camera_type}")


def load_camera_params(
    path: str | Path,
    intrinsic_file: str = 'calibration_intrinsic.json',
    extrinsic_file: str = 'calibration_extrinsic.json',
) -> List[CameraModel]:
    intrinsic_data = _load_json(path, intrinsic_file)
    extrinsic_data = _load_json(path, extrinsic_file)

    cameras: List[CameraModel] = []
    for cam_info in intrinsic_data['value0']['intrinsics']:
        model = create_camera_model(
            cam_info['camera_type'],
            cam_info['camera_name'],
            cam_info['intrinsics'],
        )
        if 'invalid_mask' in cam_info['intrinsics']:
            model.set_mask(str(path), cam_info['intrinsics']['invalid_mask'])
        cameras.append(model)

    for idx, cam_info in enumerate(extrinsic_data['value0']['extrinsics']):
        if cameras[idx].name == cam_info['camera_name']:
            cameras[idx].set_extrinsic(cam_info['extrinsics'])

    return cameras


def _load_json(path: str | Path, filename: str) -> dict:
    full = os.path.join(str(path), filename)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Calibration file not found: {full}")
    with open(full) as f:
        return json.load(f)
