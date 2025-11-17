"""Typed configuration helpers for the xDeepFM pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .utils.io import load_yaml, resolve_path


@dataclass
class DataConfig:
    dataset: Dict[str, Any]
    task: Dict[str, Any]
    features: Dict[str, Any]
    preprocess: Dict[str, Any]
    dataloader: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        raw = load_yaml(path)
        return cls(
            dataset=raw.get("dataset", {}),
            task=raw.get("task", {}),
            features=raw.get("features", {}),
            preprocess=raw.get("preprocess", {}),
            dataloader=raw.get("dataloader", {}),
        )


@dataclass
class ModelConfig:
    model: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        return cls(model=load_yaml(path).get("model", {}))


@dataclass
class TrainConfig:
    training: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        return cls(training=load_yaml(path).get("training", {}))


@dataclass
class ExperimentConfig:
    name: str
    data_config: Path
    model_config: Path
    train_config: Path
    output_dir: Path
    seed: int
    use_timestamp: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = resolve_path(path, require_exists=True, prefer_project_root=True)
        raw = load_yaml(path).get("experiment", {})
        return cls(
            name=raw.get("name", "experiment"),
            data_config=resolve_path(raw["data_config"], require_exists=True, prefer_project_root=True),
            model_config=resolve_path(raw["model_config"], require_exists=True, prefer_project_root=True),
            train_config=resolve_path(raw["train_config"], require_exists=True, prefer_project_root=True),
            output_dir=resolve_path(raw.get("output_dir", "runs"), prefer_project_root=True),
            seed=int(raw.get("seed", 42)),
            use_timestamp=bool(raw.get("use_timestamp", False)),
        )


@dataclass
class FullConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def from_experiment(cls, experiment_path: str | Path) -> "FullConfig":
        exp = ExperimentConfig.from_yaml(experiment_path)
        data_cfg = DataConfig.from_yaml(exp.data_config)
        model_cfg = ModelConfig.from_yaml(exp.model_config)
        train_cfg = TrainConfig.from_yaml(exp.train_config)
        return cls(exp, data_cfg, model_cfg, train_cfg)
