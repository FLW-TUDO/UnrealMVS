import json
import yaml
from typing import Union, Any


class Config:
    """YAML configuration with attribute-style access and optional override support."""

    def __init__(self, config_data: Union[dict, list, Any]):
        if isinstance(config_data, dict):
            object.__setattr__(self, '_config_data', {
                k: Config(v) if isinstance(v, (dict, list)) else v
                for k, v in config_data.items()
            })
        elif isinstance(config_data, list):
            object.__setattr__(self, '_config_data', [
                Config(item) if isinstance(item, dict) else item
                for item in config_data
            ])
        else:
            object.__setattr__(self, '_config_data', config_data)

    def __getattr__(self, name: str) -> Union['Config', Any]:
        data = object.__getattribute__(self, '_config_data')
        if isinstance(data, dict):
            return data.get(name)
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        data = object.__getattribute__(self, '_config_data')
        if isinstance(data, dict):
            data[name] = value
        else:
            raise AttributeError("Cannot set attribute on a non-dict Config")

    def __getitem__(self, index: int) -> Union['Config', Any]:
        data = object.__getattribute__(self, '_config_data')
        if isinstance(data, list):
            return data[index]
        raise TypeError("Config is not subscriptable (not a list)")

    def __iter__(self):
        data = object.__getattribute__(self, '_config_data')
        if isinstance(data, list):
            return iter(data)
        raise TypeError("Config is not iterable (not a list)")

    def __len__(self) -> int:
        data = object.__getattribute__(self, '_config_data')
        if isinstance(data, (list, dict)):
            return len(data)
        raise TypeError("Config has no length")

    def to_dict(self) -> Union[dict, list, Any]:
        data = object.__getattribute__(self, '_config_data')
        if isinstance(data, dict):
            return {k: v.to_dict() if isinstance(v, Config) else v for k, v in data.items()}
        if isinstance(data, list):
            return [item.to_dict() if isinstance(item, Config) else item for item in data]
        return data

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @staticmethod
    def from_yaml(file_path: str) -> 'Config':
        with open(file_path, 'r') as f:
            return Config(yaml.safe_load(f))
