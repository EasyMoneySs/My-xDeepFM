"""Preprocessing pipeline driven by YAML config."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

from ..config import DataConfig
from ..utils.io import dump_json, ensure_dir, resolve_path
from ..utils.logging import setup_logger
from .metadata import PreprocessMetadata, SplitInfo


LOGGER = setup_logger("xdeepfm.preprocess")
SECONDS_PER_DAY = 24 * 3600
_GROUP_PLACEHOLDER = "__MISSING_GROUP__"


def _normalize_column_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        lowered = stripped.lower()
        if lowered in {"none", "null"}:
            return None
        return stripped
    return value


def _load_pre_split_frames(config: DataConfig) -> Optional[Dict[str, pd.DataFrame]]:
    """加载已经划分好的 CSV 数据（train/valid/test），如果配置未提供则返回 None。"""
    dataset_cfg = config.dataset
    split_paths = dataset_cfg.get("split_paths") or dataset_cfg.get("splits")
    if not split_paths:
        return None
    csv_cfg = dataset_cfg.get("csv_format", {})
    delimiter = csv_cfg.get("delimiter", ",")
    header = 0 if csv_cfg.get("header", True) else None
    column_names = csv_cfg.get("column_names")
    read_kwargs = {}
    if header is None and column_names:
        read_kwargs["names"] = column_names

    def _read_one(name: str, raw_path: str) -> pd.DataFrame:
        path = resolve_path(raw_path, require_exists=True, prefer_project_root=True)
        LOGGER.info("Loading %s split from %s", name, path)
        return pd.read_csv(path, sep=delimiter, header=header, **read_kwargs)

    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        raw_path = split_paths.get(split_name)
        if raw_path:
            splits[split_name] = _read_one(split_name, raw_path)
    if not splits:
        raise ValueError("split_paths provided but no valid split files found")
    return splits


def _load_dataframe(config: DataConfig) -> pd.DataFrame:
    dataset_cfg = config.dataset
    provider = dataset_cfg.get("provider", "huggingface").lower()
    if provider == "huggingface":
        name = dataset_cfg["name"]
        subset = dataset_cfg.get("subset") or "train"
        LOGGER.info("Loading dataset %s (%s)", name, subset)
        ds = load_dataset(
            path=name,
            split=subset,
            cache_dir=config.preprocess.get("cache_dir"),
        )
        frame = ds.to_pandas()
        LOGGER.info("Dataset loaded with %d rows", len(frame))
        return frame
    if provider == "local_csv":
        raw_path = dataset_cfg.get("path", "")
        if not raw_path:
            raise ValueError("Local dataset requires 'path' in config.dataset")
        path = resolve_path(raw_path, require_exists=True, prefer_project_root=True)
        csv_cfg = dataset_cfg.get("csv_format", {})
        delimiter = csv_cfg.get("delimiter", ",")
        header = 0 if csv_cfg.get("header", True) else None
        column_names = csv_cfg.get("column_names")
        read_kwargs = {}
        if header is None and column_names:
            read_kwargs["names"] = column_names
        LOGGER.info("Loading local dataset %s", path)
        frame = pd.read_csv(path, sep=delimiter, header=header, **read_kwargs)
        multi_cfg = csv_cfg.get("multi_value_columns", {})
        temp_lists: Dict[str, Tuple[pd.Series, Dict]] = {}
        for col, col_cfg in multi_cfg.items():
            if col not in frame:
                continue
            sep = col_cfg.get("separator", "|")
            max_tokens = col_cfg.get("max_tokens")
            tokens = (
                frame[col]
                .fillna("")
                .astype(str)
                .str.split(sep)
                .apply(lambda values: [v for v in values if v] if isinstance(values, list) else [])
            )
            if max_tokens:
                tokens = tokens.apply(lambda values: values[: int(max_tokens)])
            temp_lists[col] = (tokens, col_cfg)
        reader_cfg = {}
        reader_cfg.update(dataset_cfg.get("reader", {}))
        reader_cfg.update(config.preprocess.get("reader", {}))
        date_col = reader_cfg.get("date_column", "date")
        if reader_cfg.get("parse_date") and date_col in frame:
            date_format = reader_cfg.get("date_format", "%Y%m%d")
            parsed = pd.to_datetime(frame[date_col].astype(str), format=date_format, errors="coerce")
            frame[date_col] = parsed
            frame["visit_hour"] = parsed.dt.hour.fillna(0).astype(int)
            frame["visit_weekday"] = parsed.dt.weekday.fillna(0).astype(int)
        rating_col = dataset_cfg.get("rating_column", "rating")
        if rating_col in frame:
            fill_rating = reader_cfg.get("fill_rating")
            drop_invalid = bool(reader_cfg.get("drop_invalid_label", False))
            frame[rating_col] = pd.to_numeric(frame[rating_col], errors="coerce")
            if drop_invalid:
                before = len(frame)
                frame = frame[frame[rating_col].notna()].copy()
                LOGGER.info(
                    "Dropped %d rows with missing %s", before - len(frame), rating_col
                )
            if fill_rating is not None:
                frame[rating_col] = frame[rating_col].fillna(float(fill_rating))
            frame[rating_col] = frame[rating_col].fillna(0.0)
        category_entry = temp_lists.get("category_ids")
        if category_entry is not None:
            category_tokens, category_cfg = category_entry
            frame["category_count"] = category_tokens.apply(len).astype(int)
            frame["primary_category"] = category_tokens.apply(
                lambda values: int(values[0]) if values and str(values[0]).isdigit() else -1
            )
            max_slots = int(category_cfg.get("max_tokens", 0) or 0)
            for idx in range(max_slots):
                col_name = f"category_id_{idx + 1}"
                frame[col_name] = category_tokens.apply(
                    lambda values, i=idx: int(values[i]) if len(values) > i and str(values[i]).isdigit() else -1
                )
        for col in ["user_id", "item_id"]:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
        LOGGER.info("Local dataset loaded with %d rows", len(frame))
        return frame
    raise ValueError(f"Unsupported dataset provider: {provider}")


def _add_time_features(frame: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    derived_cols: List[str] = []
    time_cfg = cfg.get("time_features", {})
    if not time_cfg.get("enabled"):
        return frame, derived_cols
    columns = time_cfg.get("columns", [])
    derived = time_cfg.get("derived_features", [])
    base = frame.copy()
    for feature in derived:
        if feature == "delta_conversion_time" and {"conversion_timestamp", "timestamp"}.issubset(base.columns):
            base[feature] = base["conversion_timestamp"].fillna(-1) - base["timestamp"].fillna(0)
            derived_cols.append(feature)
        elif feature == "day_index" and "timestamp" in columns and "timestamp" in base:
            base[feature] = (base["timestamp"].fillna(0) // SECONDS_PER_DAY).astype(np.int64)
            derived_cols.append(feature)
        elif feature == "hour_in_day" and "timestamp" in columns and "timestamp" in base:
            base[feature] = ((base["timestamp"].fillna(0) % SECONDS_PER_DAY) // 3600).astype(np.int64)
            derived_cols.append(feature)
    return base, derived_cols


# 构造分组列，缺失值填充为唯一 token，避免被多 split 复用
def _get_group_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Group column {column} missing for split")
    series = frame[column].astype("object").copy()
    mask = series.isna()
    if mask.any():
        series.loc[mask] = [f"{_GROUP_PLACEHOLDER}_{idx}" for idx in series.index[mask]]
    return series


def _temporal_split(frame: pd.DataFrame, split_cfg: Dict) -> Dict[str, pd.DataFrame]:
    time_col = split_cfg.get("time_column")
    if time_col not in frame:
        raise ValueError(f"Time column {time_col} missing for temporal split")
    percents = split_cfg.get("time_split_percentiles", {})
    train_ratio = float(percents.get("train", 0.7))
    valid_ratio = float(percents.get("valid", 0.15))
    cumulative_train = train_ratio
    cumulative_valid = train_ratio + valid_ratio
    frame = frame.sort_values(time_col).reset_index(drop=True)
    train_cut = frame[time_col].quantile(min(cumulative_train, 0.999))
    valid_cut = frame[time_col].quantile(min(cumulative_valid, 0.999))
    group_col = _normalize_column_name(split_cfg.get("group_column"))
    # 基于 uid/campaign 等分组，确保同一组在时间切分后不会跨 split
    if group_col:
        group_series = _get_group_series(frame, group_col)
        grouped_times = pd.DataFrame({"group": group_series, "time": frame[time_col]})
        # 使用各组最早出现时间来决定分派的时间段
        group_times = grouped_times.groupby("group")["time"].min()
        train_groups = set(group_times[group_times <= train_cut].index)
        valid_groups = set(group_times[(group_times > train_cut) & (group_times <= valid_cut)].index)
        test_groups = set(group_times[group_times > valid_cut].index)
        train_mask = group_series.isin(train_groups)
        valid_mask = group_series.isin(valid_groups)
        test_mask = group_series.isin(test_groups)
        return {
            "train": frame[train_mask],
            "valid": frame[valid_mask],
            "test": frame[test_mask],
        }
    train_mask = frame[time_col] <= train_cut
    valid_mask = (frame[time_col] > train_cut) & (frame[time_col] <= valid_cut)
    train_df = frame[train_mask]
    valid_df = frame[valid_mask]
    test_df = frame[~(train_mask | valid_mask)]
    return {"train": train_df, "valid": valid_df, "test": test_df}


def _random_split(frame: pd.DataFrame, split_cfg: Dict) -> Dict[str, pd.DataFrame]:
    ratios = split_cfg.get("random", {})
    train_ratio = ratios.get("train_ratio", 0.8)
    valid_ratio = ratios.get("valid_ratio", 0.1)
    seed = ratios.get("seed", 42)
    rng = np.random.default_rng(seed)
    group_col = _normalize_column_name(split_cfg.get("group_column"))
    if group_col:
        # 先针对 group 产生随机数，再映射到原始行，保证同组不拆分
        group_series = _get_group_series(frame, group_col)
        unique_groups = pd.Index(group_series.drop_duplicates())
        probs = pd.Series(rng.random(len(unique_groups)), index=unique_groups)
        train_cut = train_ratio
        valid_cut = train_ratio + valid_ratio
        train_groups = set(probs[probs <= train_cut].index)
        valid_groups = set(probs[(probs > train_cut) & (probs <= valid_cut)].index)
        test_groups = set(probs[probs > valid_cut].index)
        train_mask = group_series.isin(train_groups)
        valid_mask = group_series.isin(valid_groups)
        test_mask = group_series.isin(test_groups)
        return {
            "train": frame[train_mask],
            "valid": frame[valid_mask],
            "test": frame[test_mask],
        }
    probs = rng.random(len(frame))
    train_cut = train_ratio
    valid_cut = train_ratio + valid_ratio
    train_df = frame[probs <= train_cut]
    valid_df = frame[(probs > train_cut) & (probs <= valid_cut)]
    test_df = frame[probs > valid_cut]
    return {"train": train_df, "valid": valid_df, "test": test_df}


def _split_frame(frame: pd.DataFrame, config: DataConfig) -> Dict[str, pd.DataFrame]:
    split_cfg = config.preprocess.get("split", {})
    strategy = split_cfg.get("strategy", "time")
    if strategy == "time":
        splits = _temporal_split(frame, split_cfg)
    elif strategy == "random":
        splits = _random_split(frame, split_cfg)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    dedup_col = _normalize_column_name(split_cfg.get("dedup_column"))
    if dedup_col:
        priority = split_cfg.get("dedup_priority") or ["train", "valid", "test"]
        splits = _deduplicate_splits(splits, dedup_col, priority)
    group_col = _normalize_column_name(split_cfg.get("group_column"))
    if group_col:
        # 再次确认各 split 间没有重复 group
        _ensure_group_disjoint(splits, group_col)
    return splits


def _ensure_group_disjoint(splits: Dict[str, pd.DataFrame], group_col: str) -> None:
    # 如果同一 group 在多个 split 中出现，直接抛错提醒调整配置
    owners: Dict[str, str] = {}
    for split_name, df in splits.items():
        series = _get_group_series(df, group_col)
        for value in series.unique():
            if value in owners:
                raise ValueError(
                    f"Group '{value}' appears in both '{owners[value]}' and '{split_name}'. "
                    "Please adjust split configuration."
                )
            owners[value] = split_name


def _deduplicate_splits(splits: Dict[str, pd.DataFrame], column: str, priority: List[str]) -> Dict[str, pd.DataFrame]:
    """按照指定字段在 split 间去重：优先保留 priority 顺序中靠前的数据."""
    seen: set = set()
    sanitized_priority = [name for name in priority if name in splits]
    if not sanitized_priority:
        sanitized_priority = list(splits.keys())
    for split_name in sanitized_priority:
        df = splits[split_name]
        if column not in df:
            raise ValueError(f"Column '{column}' missing in split '{split_name}' for deduplication")
        series = _get_group_series(df, column)
        mask = ~series.isin(seen)
        dropped = len(df) - mask.sum()
        if dropped:
            LOGGER.info("Removed %d duplicate rows from %s split based on %s", dropped, split_name, column)
        splits[split_name] = df[mask]
        seen.update(series[mask].tolist())
    return splits


def _build_vocabs(train_df: pd.DataFrame, categorical_fields: List[str], cat_cfg: Dict, output_dir: Path) -> Dict[str, Dict[str, int]]:
    vocabs = {}
    min_freq = int(cat_cfg.get("min_freq", 1))
    vocab_dir = ensure_dir(output_dir / "vocabs")
    for field in categorical_fields:
        counts = train_df[field].fillna(cat_cfg.get("oov_token", "__OOV__")).value_counts()
        vocab = {str(value): idx + 1 for idx, (value, count) in enumerate(counts.items()) if count >= min_freq}
        dump_json(vocab, vocab_dir / f"{field}.json")
        vocabs[field] = vocab
    return vocabs


def _encode_categoricals(frame: pd.DataFrame, field: str, vocab: Dict[str, int], oov_id: int = 0) -> None:
    if field not in frame:
        frame[field] = oov_id
    frame[field] = frame[field].map(lambda x: vocab.get(str(x), oov_id)).astype(np.int64)


def _standardize(frame_dict: Dict[str, pd.DataFrame], numerical_fields: List[str], fill_value: float, scaler_type: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    train_df = frame_dict["train"]
    if not numerical_fields:
        return stats
    train_df[numerical_fields] = train_df[numerical_fields].fillna(fill_value)
    for df in frame_dict.values():
        df[numerical_fields] = df[numerical_fields].fillna(fill_value)
    if scaler_type == "none":
        return stats
    means = train_df[numerical_fields].mean()
    stds = train_df[numerical_fields].std().replace(0, 1.0)
    for col in numerical_fields:
        stats[col] = {"mean": float(means[col]), "std": float(stds[col])}
    for df in frame_dict.values():
        df[numerical_fields] = (df[numerical_fields] - means) / stds
    return stats


def run_preprocess(config_path: str | Path, force: bool = False) -> PreprocessMetadata:
    config = DataConfig.from_yaml(config_path)
    base_dir = Path(config_path).resolve().parent
    output_dir = config.preprocess.get("output_dir", "data/processed")
    output_dir = ensure_dir((base_dir / output_dir) if not Path(output_dir).is_absolute() else Path(output_dir))
    metadata_path = output_dir / "metadata.yaml"
    if metadata_path.exists() and not force:
        LOGGER.info("Metadata already exists at %s, skipping", metadata_path)
        return PreprocessMetadata.load(metadata_path)

    num_proc_cfg = config.preprocess.get("numerical_processing", {})
    provided_splits = _load_pre_split_frames(config)
    if provided_splits is not None:
        splits = {}
        derived_features = set()
        for name, df in provided_splits.items():
            processed_df, derived = _add_time_features(df, num_proc_cfg)
            splits[name] = processed_df
            derived_features.update(derived)
        derived = list(derived_features)
        split_cfg = config.preprocess.get("split", {})
        dedup_col = _normalize_column_name(split_cfg.get("dedup_column"))
        if dedup_col:
            priority = split_cfg.get("dedup_priority") or ["train", "valid", "test"]
            splits = _deduplicate_splits(splits, dedup_col, priority)
        group_col = _normalize_column_name(split_cfg.get("group_column"))
        if group_col:
            _ensure_group_disjoint(splits, group_col)
    else:
        frame = _load_dataframe(config)
        frame, derived = _add_time_features(frame, num_proc_cfg)
        splits = _split_frame(frame, config)

    categorical_fields = list(config.features.get("categorical", []))
    numerical_fields = list(config.features.get("numerical", [])) + derived
    label_column = config.task.get("label_column", "attribution")

    max_sample = int(config.preprocess.get("max_sample", 0) or 0)
    max_sample_ratio = float(config.preprocess.get("max_sample_ratio", 0) or 0.0)
    if max_sample_ratio > 0 and max_sample_ratio >= 1:
        LOGGER.warning("max_sample_ratio should be in (0,1); got %.3f, will skip ratio sampling", max_sample_ratio)
        max_sample_ratio = 0.0

    if max_sample > 0 or max_sample_ratio > 0:
        sample_seed = int(config.preprocess.get("max_sample_seed", 42))
        rng = np.random.default_rng(sample_seed)
        sampled_splits: Dict[str, pd.DataFrame] = {}
        for name, df in splits.items():
            target = len(df)
            if max_sample_ratio > 0:
                target = max(1, int(len(df) * max_sample_ratio))
            if max_sample > 0:
                target = min(target, max_sample)
            if target < len(df):
                idx = rng.choice(len(df), size=target, replace=False)
                sampled = df.iloc[idx].reset_index(drop=True)
                LOGGER.info(
                    "Downsampled %s split from %d to %d rows (ratio=%.4f, max_sample=%d, seed=%d)",
                    name,
                    len(df),
                    target,
                    max_sample_ratio,
                    max_sample,
                    sample_seed,
                )
                sampled_splits[name] = sampled
            else:
                sampled_splits[name] = df.reset_index(drop=True)
        splits = sampled_splits
    else:
        splits = {name: df.reset_index(drop=True) for name, df in splits.items()}

    vocabs = _build_vocabs(splits["train"], categorical_fields, config.preprocess.get("categorical_encoding", {}), output_dir)
    vocab_sizes = {}
    for field in categorical_fields:
        vocab = vocabs.get(field, {})
        for split_df in splits.values():
            _encode_categoricals(split_df, field, vocab)
        vocab_sizes[field] = len(vocab) + 1

    num_proc_cfg = config.preprocess.get("numerical_processing", {})
    scaler_stats = _standardize(splits, numerical_fields, num_proc_cfg.get("fill_na", 0.0), num_proc_cfg.get("scaler", "standard"))

    split_infos: Dict[str, SplitInfo] = {}
    base_cols = categorical_fields + numerical_fields
    if label_column not in base_cols:
        selected_cols = base_cols + [label_column]
    else:
        selected_cols = base_cols
    for name, df in splits.items():
        split_path = output_dir / f"{name}.parquet"
        df_to_save = df[selected_cols].copy()
        df_to_save[label_column] = df_to_save[label_column].astype(np.float32)
        df_to_save.to_parquet(split_path, index=False)
        split_infos[name] = SplitInfo(name=name, path=split_path.resolve(), num_rows=len(df))
        LOGGER.info("Saved %s split to %s (%d rows)", name, split_path, len(df))

    metadata = PreprocessMetadata(
        label_column=label_column,
        categorical_fields=categorical_fields,
        numerical_fields=numerical_fields,
        vocab_sizes=vocab_sizes,
        scaler_stats=scaler_stats,
        splits=split_infos,
        extras={
            "derived_numerical": derived,
            "config_path": str(Path(config_path).resolve()),
        },
    )
    metadata.save(metadata_path)
    return metadata
