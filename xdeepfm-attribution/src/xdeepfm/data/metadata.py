"""Metadata helpers for processed datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..utils.io import load_yaml, dump_yaml


@dataclass
class SplitInfo:
    name: str
    path: Path
    num_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {"path": str(self.path), "num_rows": self.num_rows}


@dataclass
class PreprocessMetadata:
    label_column: str
    categorical_fields: List[str]
    numerical_fields: List[str]
    vocab_sizes: Dict[str, int]
    scaler_stats: Dict[str, Dict[str, float]]
    splits: Dict[str, SplitInfo] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_column": self.label_column,
            "categorical_fields": self.categorical_fields,
            "numerical_fields": self.numerical_fields,
            "vocab_sizes": self.vocab_sizes,
            "scaler_stats": self.scaler_stats,
            "splits": {name: split.to_dict() for name, split in self.splits.items()},
            "extras": self.extras,
        }

    def save(self, path: str | Path) -> None:
        dump_yaml(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessMetadata":
        raw = load_yaml(path)
        splits = {
            name: SplitInfo(name=name, path=Path(info["path"]).resolve(), num_rows=info.get("num_rows", 0))
            for name, info in raw.get("splits", {}).items()
        }
        return cls(
            label_column=raw.get("label_column", "attribution"),
            categorical_fields=list(raw.get("categorical_fields", [])),
            numerical_fields=list(raw.get("numerical_fields", [])),
            vocab_sizes=dict(raw.get("vocab_sizes", {})),
            scaler_stats=dict(raw.get("scaler_stats", {})),
            splits=splits,
            extras=dict(raw.get("extras", {})),
        )
