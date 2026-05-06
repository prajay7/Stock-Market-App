from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelRegistry:
    model_dir: Path
    keep_last_versions: int = 0

    def save_model(self, model: Any, model_name: str, version: str) -> Path:
        latest_path = self.model_dir / f"{model_name}_latest.joblib"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, latest_path)
        if int(self.keep_last_versions) > 0:
            versioned_path = self.model_dir / f"{model_name}_{version}.joblib"
            shutil.copy2(latest_path, versioned_path)
        return latest_path

    def save_metadata(self, model_name: str, version: str, metadata: dict[str, Any]) -> Path:
        payload = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        latest_path = self.model_dir / f"{model_name}_latest.metadata.json"
        latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if int(self.keep_last_versions) > 0:
            versioned_path = self.model_dir / f"{model_name}_{version}.metadata.json"
            versioned_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return latest_path

    @staticmethod
    def _version_from_artifact(path: Path, model_name: str, suffix: str) -> str | None:
        name = path.name
        prefix = f"{model_name}_"
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        version = name[len(prefix) : -len(suffix)]
        if re.fullmatch(r"\d{14}", version):
            return version
        return None

    def prune_old_versions(self, model_name: str, keep_last: int | None = None) -> list[Path]:
        keep_count = max(0, int(keep_last if keep_last is not None else self.keep_last_versions))
        suffixes = [".joblib", ".metadata.json", ".feature_importance.json"]
        versions: set[str] = set()

        for suffix in suffixes:
            for path in self.model_dir.glob(f"{model_name}_*{suffix}"):
                version = self._version_from_artifact(path, model_name, suffix)
                if version is not None:
                    versions.add(version)

        versions_to_delete = sorted(versions) if keep_count == 0 else sorted(versions)[:-keep_count]
        deleted: list[Path] = []
        for version in versions_to_delete:
            for suffix in suffixes:
                path = self.model_dir / f"{model_name}_{version}{suffix}"
                if path.exists():
                    path.unlink()
                    deleted.append(path)
        return deleted

    def list_models(self) -> list[str]:
        return sorted([p.name for p in self.model_dir.glob("*.joblib")])
