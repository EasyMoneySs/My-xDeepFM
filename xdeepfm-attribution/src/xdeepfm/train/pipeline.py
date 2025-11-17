"""Training and evaluation pipeline."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ..config import FullConfig, DataConfig
from ..data.dataset import build_dataloader
from ..data.metadata import PreprocessMetadata
from ..data.preprocess import run_preprocess
from ..model.xdeepfm import build_model
from ..utils.io import dump_yaml, ensure_dir, resolve_path
from ..utils.logging import setup_logger
from ..utils.seed import set_seed

LOGGER = setup_logger("xdeepfm.train")


def _resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _get_metadata_path(data_cfg: DataConfig, data_cfg_path: Path) -> Path:
    output_dir = data_cfg.preprocess.get("output_dir", "data/processed")
    return _resolve_path(output_dir, data_cfg_path.parent) / "metadata.yaml"


def _load_or_create_metadata(full_cfg: FullConfig) -> PreprocessMetadata:
    metadata_path = _get_metadata_path(full_cfg.data, full_cfg.experiment.data_config)
    if metadata_path.exists():
        LOGGER.info("Loading cached metadata from %s", metadata_path)
        return PreprocessMetadata.load(metadata_path)
    LOGGER.info("Metadata missing, running preprocessing pipeline")
    return run_preprocess(full_cfg.experiment.data_config)


def _batch_value(value, split: str, default: int) -> int:
    if isinstance(value, dict):
        return int(value.get(split, value.get("default", default)))
    return int(value or default)


def _build_dataloaders(metadata: PreprocessMetadata, data_cfg: DataConfig) -> Dict[str, DataLoader]:
    loaders = {}
    batch_cfg = data_cfg.dataloader.get("batch_size", 1024)
    num_workers = int(data_cfg.dataloader.get("num_workers", 0))
    pin_memory = bool(data_cfg.dataloader.get("pin_memory", False))
    shuffle_train = bool(data_cfg.dataloader.get("shuffle_train", True))
    for split in ["train", "valid", "test"]:
        loaders[split] = build_dataloader(
            split=split,
            metadata=metadata,
            batch_size=_batch_value(batch_cfg, split, 1024),
            num_workers=num_workers,
            shuffle=shuffle_train if split == "train" else False,
            pin_memory=pin_memory,
        )
    return loaders


def _resolve_device(name: str) -> torch.device:
    name = (name or "cpu").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError:
        return float("nan")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    task_type: str = "binary_classification",
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    labels_collected: list[np.ndarray] = []
    preds_collected: list[np.ndarray] = []
    num_batches = len(loader)
    last_batch_idx = 0
    is_regression = task_type == "regression"
    # Iterate over loader once; optimizer is only used when training.
    for batch_idx, batch in enumerate(tqdm(loader, desc="train" if train_mode else "eval", leave=False), 1):
        batch = _move_batch(batch, device)
        logits = model(batch)
        labels = batch["label"].squeeze(-1)
        loss = criterion(logits, labels)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        labels_collected.append(labels.detach().cpu().numpy())
        if is_regression:
            preds_collected.append(logits.detach().cpu().numpy())
        else:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds_collected.append(probs)
        # if batch_idx % 100 == 0 :
        #     LOGGER.info(f"Batch {batch_idx}/{num_batches}  Avg Loss: {total_loss / (batch_idx * loader.batch_size)},AUC: {_safe_auc(np.concatenate(labels_collected).reshape(-1), np.concatenate(preds_collected).reshape(-1))}")
        last_batch_idx = batch_idx
    if not labels_collected:
        return {"mse": 0.0, "mae": 0.0} if is_regression else {"logloss": 0.0, "auc": float("nan")}
    labels_np = np.concatenate(labels_collected).reshape(-1)
    preds_np = np.concatenate(preds_collected).reshape(-1)
    avg_loss = total_loss / len(loader.dataset)
    if is_regression:
        # Regression reports MSE/MAE to align with task metrics.
        mse = float(np.mean((preds_np - labels_np) ** 2))
        mae = float(np.mean(np.abs(preds_np - labels_np)))
        # print(f"Batch {last_batch_idx}/{num_batches}  Avg Loss: {avg_loss}, MSE: {mse}, MAE: {mae}")
        return {"mse": mse, "mae": mae}
    auc = _safe_auc(labels_np, preds_np)
    # print(f"Batch {last_batch_idx}/{num_batches}  Avg Loss: {avg_loss}, AUC: {auc}")
    return {"logloss": float(avg_loss), "auc": auc}


class EarlyStopping:
    def __init__(self, metric: str, mode: str, patience: int, delta: float) -> None:
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.best_score: float | None = None
        self.num_bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return value < self.best_score - self.delta
        return value > self.best_score + self.delta

    def update(self, value: float) -> bool:
        if self._is_improvement(value):
            self.best_score = value
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.num_bad_epochs >= self.patience if self.patience > 0 else False


def _get_metric(metrics: Dict[str, Dict[str, float]], key: str) -> float:
    split, metric = key.split("_")
    return metrics[split][metric]


def _resolve_checkpoint_dir(save_dir: str | None, default_base: Path) -> Path:
    if save_dir:
        return ensure_dir(resolve_path(save_dir, prefer_project_root=True))
    return ensure_dir(default_base / "checkpoints")


def _resolve_resume_path(resume_value: str, ckpt_dir: Path) -> Path:
    candidate = Path(resume_value)
    if not candidate.is_absolute():
        relative = (ckpt_dir / candidate).resolve()
        if relative.exists():
            return relative
    return resolve_path(candidate, require_exists=True, prefer_project_root=True)


def _checkpoint_payload(
    model_state: Dict[str, torch.Tensor],
    optimizer_state: Dict[str, torch.Tensor],
    epoch: int,
    history: List[Dict[str, Dict[str, float]]],
    monitor_key: str,
    best_state: Dict[str, torch.Tensor] | None,
    best_optimizer_state: Dict[str, torch.Tensor] | None,
    early_stopper: EarlyStopping,
    metrics: Dict[str, Dict[str, float]] | None = None,
) -> Dict:
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "history": list(history),
        "monitor_key": monitor_key,
        "best_model_state": best_state or model_state,
        "best_optimizer_state": best_optimizer_state,
        "early_stopping": {
            "best_score": early_stopper.best_score,
            "num_bad_epochs": early_stopper.num_bad_epochs,
        },
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def _epoch_ckpt_sort_key(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


def _prune_old_checkpoints(managed: List[Path], max_keep: int) -> None:
    if max_keep <= 0:
        return
    while len(managed) > max_keep:
        victim = managed.pop(0)
        try:
            victim.unlink(missing_ok=True)
        except FileNotFoundError:
            continue


def train_experiment(experiment_path: str | Path, use_timestamp: bool | None = None) -> Path:
    full_cfg = FullConfig.from_experiment(experiment_path)
    set_seed(full_cfg.experiment.seed)

    # 优先使用函数入参，其次读取实验配置中的 use_timestamp
    ts_enabled = use_timestamp if use_timestamp is not None else bool(getattr(full_cfg.experiment, "use_timestamp", False))

    base_dir = Path(full_cfg.experiment.output_dir)
    if ts_enabled:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_dir(base_dir.parent / f"{base_dir.name}_{ts}")
    else:
        output_dir = ensure_dir(base_dir)

    # Setup file-based logging for the experiment run
    global LOGGER
    LOGGER = setup_logger("xdeepfm.train", log_file=output_dir / "training.log")

    # Save the full configuration to the output directory for reproducibility
    config_log_path = output_dir / "config_log.yaml"
    config_to_save = {
        "experiment": {
            "name": full_cfg.experiment.name,
            "data_config": str(full_cfg.experiment.data_config),
            "model_config": str(full_cfg.experiment.model_config),
            "train_config": str(full_cfg.experiment.train_config),
            "output_dir": str(output_dir),
            "seed": full_cfg.experiment.seed,
        },
        "data": asdict(full_cfg.data),
        "model": asdict(full_cfg.model),
        "train": asdict(full_cfg.train),
    }
    dump_yaml(config_to_save, config_log_path)
    LOGGER.info("Saved full configuration to %s", config_log_path)

    metadata = _load_or_create_metadata(full_cfg)
    loaders = _build_dataloaders(metadata, full_cfg.data)

    device = _resolve_device(full_cfg.train.training.get("device", "cpu"))
    model = build_model(metadata, full_cfg.model.model).to(device)

    training_cfg = full_cfg.train.training
    ckpt_cfg = training_cfg.get("checkpoint", {})
    interval_value = int(ckpt_cfg.get("save_interval", 1) or 0)
    save_interval = interval_value if interval_value > 0 else None
    max_ckpt_keep = max(0, int(ckpt_cfg.get("max_to_keep", 0)))
    task_type = full_cfg.data.task.get("type", "binary_classification").lower()
    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    opt_cfg = training_cfg.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    if opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.0))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    default_monitor = "valid_mse" if task_type == "regression" else "valid_logloss"
    monitor_key = early_cfg.get("metric") or default_monitor
    early_stopper = EarlyStopping(
        metric=monitor_key,
        mode=early_cfg.get("mode", "min"),
        patience=int(early_cfg.get("patience", 2)),
        delta=float(early_cfg.get("delta", 0.0)),
    )

    ckpt_dir = _resolve_checkpoint_dir(ckpt_cfg.get("save_dir"), output_dir)
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    metrics_path = output_dir / "metrics_test.yaml"
    managed_epoch_ckpts: List[Path] = sorted(
        [p for p in ckpt_dir.glob("epoch_*.pt") if p.is_file()],
        key=_epoch_ckpt_sort_key,
    )
    _prune_old_checkpoints(managed_epoch_ckpts, max_ckpt_keep)

    history: list[Dict[str, Dict[str, float]]] = []
    best_state = None
    best_optimizer_state = None
    start_epoch = 1

    resume_target = ckpt_cfg.get("resume_from")
    if resume_target:
        resume_path = _resolve_resume_path(str(resume_target), ckpt_dir)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        last_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        monitor_key = checkpoint.get("monitor_key", monitor_key)
        early_state = checkpoint.get("early_stopping", {})
        if "best_score" in early_state:
            early_stopper.best_score = early_state["best_score"]
        if "num_bad_epochs" in early_state:
            early_stopper.num_bad_epochs = int(early_state["num_bad_epochs"])
        best_state = checkpoint.get("best_model_state", checkpoint.get("model_state"))
        best_optimizer_state = checkpoint.get("best_optimizer_state")
        LOGGER.info("Resumed training from %s (epoch %d)", resume_path, last_epoch)

    total_epochs = int(training_cfg.get("epochs", 1))
    if start_epoch > total_epochs:
        LOGGER.warning(
            "Resume epoch %d exceeds configured epochs (%d); skipping further training",
            start_epoch - 1,
            total_epochs,
        )

    for epoch in range(start_epoch, total_epochs + 1):
        LOGGER.info("Epoch %d", epoch)
        train_metrics = _run_epoch(model, loaders["train"], criterion, device, optimizer, task_type)
        LOGGER.info("Train metrics: %s", train_metrics)
        valid_metrics = _run_epoch(model, loaders["valid"], criterion, device, task_type=task_type)
        LOGGER.info("Valid metrics: %s", valid_metrics)
        metrics_bundle = {"train": train_metrics, "valid": valid_metrics}
        history.append({"epoch": epoch, **metrics_bundle})
        dump_yaml({"test": None, "history": history}, metrics_path)
        try:
            monitor_value = _get_metric(metrics_bundle, monitor_key)
        except KeyError:
            monitor_key = default_monitor
            monitor_value = _get_metric(metrics_bundle, monitor_key)
        should_stop = False
        if early_stopper.update(monitor_value):
            best_state = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
            best_payload = _checkpoint_payload(
                best_state,
                best_optimizer_state,
                epoch,
                history,
                monitor_key,
                best_state,
                best_optimizer_state,
                early_stopper,
                metrics_bundle,
            )
            torch.save(best_payload, best_ckpt)
            LOGGER.info("Saved new best checkpoint at epoch %d", epoch)
        elif early_enabled and early_stopper.should_stop():
            LOGGER.info("Early stopping triggered at epoch %d", epoch)
            should_stop = True

        current_payload = _checkpoint_payload(
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            history,
            monitor_key,
            best_state,
            best_optimizer_state,
            early_stopper,
            metrics_bundle,
        )
        torch.save(current_payload, last_ckpt)
        if save_interval and epoch % save_interval == 0:
            epoch_ckpt = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(current_payload, epoch_ckpt)
            managed_epoch_ckpts.append(epoch_ckpt)
            _prune_old_checkpoints(managed_epoch_ckpts, max_ckpt_keep)
        if should_stop:
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    test_metrics, preds_df = evaluate_model(model, loaders["test"], metadata, device, task_type)
    dump_yaml({"test": test_metrics, "history": history}, metrics_path)
    preds_path = output_dir / "preds_test.parquet"
    preds_df.to_parquet(preds_path, index=False)
    LOGGER.info("Test metrics saved to %s", metrics_path)
    return best_ckpt


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    metadata: PreprocessMetadata,
    device: torch.device,
    task_type: str = "binary_classification",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    stats = _run_epoch(model, loader, criterion, device, task_type=task_type)
    split_path = metadata.splits["test"].path
    frame = pd.read_parquet(split_path).reset_index(drop=True)
    preds = []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            if task_type == "regression":
                preds.append(logits.cpu())
            else:
                preds.append(torch.sigmoid(logits).cpu())
    preds_tensor = torch.cat(preds).squeeze(-1)
    frame = frame.iloc[: preds_tensor.shape[0]].copy()
    frame["prediction"] = preds_tensor.numpy()
    keep_cols = [col for col in ["uid", "campaign", metadata.label_column, "prediction"] if col in frame.columns]
    return stats, frame[keep_cols]


def evaluate_checkpoint(experiment_path: str | Path, checkpoint_path: str | Path) -> Dict[str, float]:
    full_cfg = FullConfig.from_experiment(experiment_path)
    metadata = _load_or_create_metadata(full_cfg)
    loaders = _build_dataloaders(metadata, full_cfg.data)
    device = _resolve_device(full_cfg.train.training.get("device", "cpu"))
    model = build_model(metadata, full_cfg.model.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    task_type = full_cfg.data.task.get("type", "binary_classification").lower()
    metrics, _ = evaluate_model(model, loaders["test"], metadata, device, task_type)
    return metrics
