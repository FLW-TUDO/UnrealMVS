# helpers/

Two download scripts for preparing training data.
Run all commands from the **project root**, not from inside `helpers/`.

---

## download_unrealmvs.py

Downloads the **UnrealMVS** scenes from HuggingFace
([flw-tu-dortmund/UnrealMVS](https://huggingface.co/datasets/flw-tu-dortmund/UnrealMVS)).
Data arrives in the correct format — no conversion needed.

```bash
python helpers/download_unrealmvs.py                                    # all scenes
python helpers/download_unrealmvs.py --scene polar                      # one scene (fuzzy)
python helpers/download_unrealmvs.py --scene polar --scene factory      # multiple scenes
python helpers/download_unrealmvs.py --scene polar --split              # download + split
python helpers/download_unrealmvs.py --list                             # list available scenes
python helpers/download_unrealmvs.py --out /data/datasets               # custom output dir
```

### Available scenes

| Scene | Frames |
|---|---|
| Bazaar | 500 |
| Cambodia | 500 |
| Catacomb | 850 |
| Cigar Room | 500 |
| CitySampleDay | 3000 |
| CitySampleNight | 2000 |
| ElectricDreamsEnv | 900 |
| Engine Hall | 500 |
| FactoryEnvironment | 2000 |
| Mediterranean | 500 |
| PolarStation | 250 |
| PoliceHeadquarters | 1000 |
| Tokyo | 500 |
| WindyHouse | 500 |

### Split options

After downloading, frames must be split into `train/`, `eval/`, `test/` subdirectories
before training. Pass `--split` to do this immediately after download,
or run it separately on already-downloaded data:

```bash
python helpers/download_unrealmvs.py --scene polar --split
```

Default split: **70% train / 20% eval / 10% test** (seed 42).
Override with `--train-ratio`, `--eval-ratio`, `--seed`.

---

## download_omnimvs.py

Downloads the original **OmniMVS** datasets from Google Drive,
converts them from the raw format, and optionally splits them.

```bash
python helpers/download_omnimvs.py [subcommand] [options]
```

### Subcommands

| Command | What it does |
|---|---|
| `setup` | Download + extract + convert (default when no command is given) |
| `download` | Download and extract raw files only |
| `convert` | Convert already-downloaded raw data |
| `split` | Assign converted frames to `train/` `eval/` `test/` subdirectories |
| `verify` | Check all EXR files and print a per-split summary |
| `visualize` | Generate a camera-coverage overview PNG for one frame |

### Available datasets

| Name | Match string | Variants | Frames |
|---|---|---|---|
| OmniMVS Urban | `urban` | Urban_Sunny, Urban_Cloudy, Urban_Sunset | 1000 per variant |
| OmniMVS House | `house` | House | 2560 |
| OmniMVS Things | `things` | Things | 10240 |

### setup

```bash
python helpers/download_omnimvs.py                            # all datasets
python helpers/download_omnimvs.py setup --dataset urban      # one dataset
python helpers/download_omnimvs.py setup --split              # + create splits
python helpers/download_omnimvs.py setup --split --verify     # + verify EXR files
```

### split

```bash
python helpers/download_omnimvs.py split                      # all datasets
python helpers/download_omnimvs.py split --dataset urban
python helpers/download_omnimvs.py split --seed 42
```

Split sizes:

| Dataset | Train | Eval | Test |
|---|---|---|---|
| Urban (each variant) | 350 | 350 | 300 |
| House | 1024 | 1024 | 512 |
| Things | 4608 | 4608 | 1024 |

### verify

```bash
python helpers/download_omnimvs.py verify
python helpers/download_omnimvs.py verify --dataset urban --workers 8
```

### Output structure

After conversion each dataset directory looks like this:

```
datasets/Urban_Sunny/
├── calibration_intrinsic.json
├── calibration_extrinsic.json
├── train/
│   ├── 0001/
│   │   ├── FisheyeCamera_1.exr
│   │   ├── FisheyeCamera_2.exr
│   │   ├── FisheyeCamera_3.exr
│   │   ├── FisheyeCamera_4.exr
│   │   ├── Equirectangular_Depth.exr
│   │   └── Equirectangular.exr
│   └── ...
├── eval/
└── test/
```
