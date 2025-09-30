"""Parameter state management utilities."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

TUNABLE_KEYS: tuple[str, ...] = (
    "GEN_TEMPERATURE",
    "GEN_TOP_P",
    "GEN_TOP_K",
    "GEN_REPEAT_PENALTY",
    "GEN_PRESENCE_PENALTY",
    "GEN_FREQUENCY_PENALTY",
)


class ParameterStateManager:
    """Handles reading, updating, and backing up generation parameters."""

    def __init__(self, env_file: Path | None = None) -> None:
        self.repo_root = Path(__file__).resolve().parents[3]
        self.env_file = env_file or self.repo_root / ".env"
        self.state_dir = self.repo_root / "vega_state" / "self_optimization"
        self.backup_dir = self.state_dir / "backups"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.current_state_file = self.state_dir / "current_parameters.json"

    def _load_env(self) -> Dict[str, str]:
        if not self.env_file.exists():
            return {}
        content = self.env_file.read_text(encoding="utf-8").splitlines()
        env: Dict[str, str] = {}
        for line in content:
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            env[key.strip()] = val.strip()
        return env

    def _write_env(self, data: Mapping[str, str]) -> None:
        lines = []
        for key, value in sorted(data.items()):
            lines.append(f"{key}={value}")
        self.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def current_parameters(self) -> Dict[str, str]:
        env = self._load_env()
        return {key: env[key] for key in TUNABLE_KEYS if key in env}

    def backup_current(self) -> Path:
        snapshot = self.current_parameters()
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        backup_path = self.backup_dir / f"params_{timestamp}.json"
        backup_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        self.current_state_file.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "parameters": snapshot,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return backup_path

    def apply_parameters(self, updates: Mapping[str, float | int | str]) -> None:
        env = self._load_env()
        for key, value in updates.items():
            env[key] = str(value)
        self._write_env(env)
        self.current_state_file.write_text(
            json.dumps({"timestamp": time.time(), "parameters": env}, indent=2),
            encoding="utf-8",
        )

    def revert_last_backup(self) -> Path | None:
        backups = sorted(self.backup_dir.glob("params_*.json"))
        if not backups:
            return None
        latest = backups[-1]
        snapshot = json.loads(latest.read_text(encoding="utf-8"))
        env = self._load_env()
        env.update(snapshot)
        self._write_env(env)
        return latest

    def list_backups(self) -> Iterable[Path]:
        return sorted(self.backup_dir.glob("params_*.json"))
