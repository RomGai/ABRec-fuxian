from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def mm_encoder(self) -> Dict[str, Any]:
        return self.raw["mm_encoder"]

    @property
    def losses(self) -> Dict[str, Any]:
        return self.raw["losses"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def search_space(self) -> Dict[str, List[Any]]:
        return self.raw["search"]


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
