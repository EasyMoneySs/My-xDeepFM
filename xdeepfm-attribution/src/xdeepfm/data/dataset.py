"""PyTorch dataset/dataloader utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .metadata import PreprocessMetadata


class AttributionDataset(Dataset):
    """Simple in-memory dataset backed by Parquet files."""

    def __init__(
        self,
        data_path: str | Path,
        metadata: PreprocessMetadata,
        categorical_fields: Optional[Iterable[str]] = None,
        numerical_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.path = Path(data_path)
        self.metadata = metadata
        self.categorical_fields = list(categorical_fields or metadata.categorical_fields)
        self.numerical_fields = list(numerical_fields or metadata.numerical_fields)
        self.label_column = metadata.label_column

        frame = pd.read_parquet(self.path)
        for col in self.categorical_fields:
            if col not in frame:
                frame[col] = 0
        for col in self.numerical_fields:
            if col not in frame:
                frame[col] = 0.0
        self.categorical = torch.as_tensor(frame[self.categorical_fields].values.astype(np.int64))
        self.numerical = torch.as_tensor(frame[self.numerical_fields].values.astype(np.float32))
        self.labels = torch.as_tensor(frame[self.label_column].values.astype(np.float32)).unsqueeze(-1)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "categorical": self.categorical[idx],
            "numerical": self.numerical[idx],
            "label": self.labels[idx],
        }


def build_dataloader(
    split: str,
    metadata: PreprocessMetadata,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    split_info = metadata.splits[split]
    dataset = AttributionDataset(split_info.path, metadata)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
