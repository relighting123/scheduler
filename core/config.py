import yaml
import os
from typing import Any, Dict

class Config:
    _instance = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._config_data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config_data.get(key, default)

    @property
    def app(self) -> Dict[str, Any]:
        return self._config_data.get("app", {})

    @property
    def databases(self) -> Dict[str, Any]:
        return self._config_data.get("databases", {})

config = Config()
