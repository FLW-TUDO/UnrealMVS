import torch


def load_state_dict(path: str, device) -> dict:
    """Load a checkpoint, handling multiple formats:

    - Plain state_dict
    - {"net_state_dict": ...} (hyu-cvlab original format)
    - Legacy .bn. -> .batch_norm. key remapping
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("net_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k.replace(".bn.", ".batch_norm."): v for k, v in sd.items()}
