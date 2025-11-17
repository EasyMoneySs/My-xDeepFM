"""Utility helpers for filesystem and serialization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _candidate_paths(path: Path, prefer_project_root: bool = False) -> list[Path]:
    """Return possible absolute paths for a relative input."""
    bases = (PROJECT_ROOT, Path.cwd()) if prefer_project_root else (Path.cwd(), PROJECT_ROOT)
    seen: list[Path] = []
    for base in bases:
        candidate = (base / path).resolve()
        if candidate not in seen:
            seen.append(candidate)
    return seen


def resolve_path(
    path: str | Path,
    *,
    require_exists: bool = False,
    prefer_project_root: bool = False,
) -> Path:
    """Resolve a path relative to cwd/project root with optional existence check."""
    path = Path(path).expanduser()
    if path.is_absolute():
        if require_exists and not path.exists():
            raise FileNotFoundError(f"Could not locate '{path}'")
        return path
    candidates = _candidate_paths(path, prefer_project_root=prefer_project_root)
    if require_exists:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not locate '{path}' (tried: {tried})")
    return candidates[0]


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _expand_env(value: Any) -> Any:
    """Recursively expand environment variables inside YAML values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> Any:
    path = resolve_path(path, require_exists=True, prefer_project_root=True)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _expand_env(raw)


def dump_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
