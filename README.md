# UnrealMVS

Neural network for depth estimation from 4 fisheye cameras.
Based on [hyu-cvlab/omnimvs-pytorch](https://github.com/hyu-cvlab/omnimvs-pytorch), rewritten to work with custom data formats and training pipelines.

- **Datasets (HuggingFace):** [flw-tu-dortmund/UnrealMVS](https://huggingface.co/datasets/flw-tu-dortmund/UnrealMVS) — the datasets used in this project.
- **Data Generator:** [FLW-TUDO/UnrealMVS-DataGenerator](https://github.com/FLW-TUDO/UnrealMVS-DataGenerator) — the data generator for Unreal Engine 5 used to create the synthetic scenes.

## Installation

```bash
git clone https://github.com/KarolPiotrRoesner/UnrealMVS.git
cd UnrealMVS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Datasets

### UnrealMVS (HuggingFace)

Pre-processed scenes hosted at
[flw-tu-dortmund/UnrealMVS](https://huggingface.co/datasets/flw-tu-dortmund/UnrealMVS).
No conversion needed — download and split in one step:

```bash
python helpers/download_unrealmvs.py --scene polar --split   # one scene
python helpers/download_unrealmvs.py --split                  # all scenes
```

### OmniMVS (raw datasets)

To use the original OmniMVS datasets, use `helpers/download_omnimvs.py`.
It downloads, converts and splits the raw data into the same format as ours used during training.

```bash
python helpers/download_omnimvs.py --split   # all datasets
```

See [helpers/README.md](helpers/README.md) for full usage instructions for both scripts.

Each scene is captured with a **4-fisheye-camera rig** mounted in a fixed configuration.
Per frame the dataset provides:

| File | Description |
|---|---|
| `FisheyeCamera_1.exr` – `FisheyeCamera_4.exr` | Raw fisheye images, float32 |
| `Equirectangular_Depth.exr` | Ground-truth depth in metres, float32 (0 = invalid) |
| `Equirectangular.exr` | Equirectangular colour panorama, float32 |
| `calibration_intrinsic.json` | Per-camera intrinsics (one file per scene) |
| `calibration_extrinsic.json` | Rig extrinsics — camera-to-rig transforms |

```
FactoryEnvironment/
├── calibration_intrinsic.json
├── calibration_extrinsic.json
├── mask.png
├── 0000/
│   ├── FisheyeCamera_1.exr
│   ├── FisheyeCamera_2.exr
│   ├── FisheyeCamera_3.exr
│   ├── FisheyeCamera_4.exr
│   ├── Equirectangular_Depth.exr
│   └── Equirectangular.exr
├── 0001/
│   └── ...
└── ...
```

---

## Training & Evaluation

### Configuration

Training is controlled by a YAML config file. The default is `data/training.yaml`.
A faster variant for testing is `data/training_quick.yaml`.

Key config options:

| Field | Description |
|---|---|
| `datasets[].subsets` | List of dataset folder names under `data_root` |
| `datasets[].epochs` | Number of epochs for this training stage |
| `parameters.data.data_root` | Root directory containing the converted datasets |
| `parameters.general.device` | PyTorch device, e.g. `cuda:0` or `cpu` |
| `parameters.model.weights_path` | Checkpoint to load at the start of training |
| `parameters.logging.log_dir` | Where checkpoints and logs are saved |

Training supports multiple sequential stages — the first stage trains from scratch,
each subsequent stage finetunes the same model on a different dataset mix.

### Train

```bash
# Default config
python main.py --train

# Custom config
python main.py --train --config-file data/training.yaml

# Resume from a checkpoint
python main.py --train --config-file data/training.yaml --weights-path logs/OmniMVS_V2_latest.pt

# Override experiment name
python main.py --train --name my_experiment
```

### Evaluate

```bash
# Evaluate on the test split (default)
python main.py --evaluate --config-file data/training.yaml --weights-path logs/OmniMVS_V2_latest.pt

# Evaluate on specific splits
python main.py --evaluate --weights-path logs/OmniMVS_V2_latest.pt --splits test,eval

# Evaluate on specific datasets only
python main.py --evaluate --weights-path logs/OmniMVS_V2_latest.pt --datasets Urban_Sunny,House

# Save per-frame inverse-depth PNGs
python main.py --evaluate --weights-path logs/OmniMVS_V2_latest.pt --save-dir output/

# Use half precision for faster inference
python main.py --evaluate --weights-path logs/OmniMVS_V2_latest.pt --precision float16
```

### Train + evaluate in one run

```bash
python main.py --train --evaluate --config-file data/training.yaml
```

---

## Citation

This work was accepted for **ETFA 2026**.

If you use UnrealMVS in your research, please cite:

```bibtex
@INPROCEEDINGS{11725285,
  author={Rösner, Karol Piotr and Rutinowski, Jérôme and Surmann, Hartmut and Kirchheim, Alice},
  booktitle={2026 IEEE 31st International Conference on Emerging Technologies and Factory Automation (ETFA)}, 
  title={UnrealMVS: A Generator for Synthetic Multi-Fisheye Omnidirectional Depth Data}, 
  year={2026},
  volume={},
  number={},
  pages={01-08},
  keywords={Modeling;Cameras;Printing;Pipelines;Training;Rendering (computer graphics);Image sensors;Calibration;Conferences;Urban areas;Synthetic Datasets;Equirectangular Images;Depth Images;Multi-View Stereo;Fisheye Cameras},
  doi={10.1109/ETFA68377.2026.11725285}}
}
```

---

## Acknowledgements

This project was created in cooperation between **TU Dortmund University** and
**Westphalian University of Applied Sciences (WHS)**.

<p align="center">
  <img src="docs/assets/tu.png" height="120" alt="TU Dortmund University">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/wh.png" height="120" alt="Westphalian University of Applied Sciences">
</p>
