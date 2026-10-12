from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.models.bpn import BPNClassifier
from src.utils.config import ROOT


def new_run_dir(base: Path | None = None) -> Path:
    root = base or (ROOT / "models")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"run_{stamp}"
    suffix = 2
    while path.exists():
        path = root / f"run_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    models = serializable.pop("models", {})
    joblib.dump(serializable, run_dir / "bundle.joblib")
    joblib.dump(models, run_dir / "sklearn_models.joblib")
    bpn = payload.get("bpn")
    if bpn is not None:
        joblib.dump(bpn.save_weights(), run_dir / "bpn_weights.joblib")
    metrics = {
        name: values.as_dict() if hasattr(values, "as_dict") else values
        for name, values in payload.get("metrics", {}).items()
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    meta = {
        "dataset": payload.get("dataset_name"),
        "seed": payload.get("seed"),
        "selected_features": payload.get("selected_features"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": payload.get("mode"),
        "training_seconds": payload.get("duration"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    latest = run_dir.parent / "latest.json"
    latest.write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")


def load_latest(base: Path | None = None) -> dict[str, Any] | None:
    root = base or (ROOT / "models")
    pointer = root / "latest.json"
    if not pointer.exists():
        return None
    run_dir = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])
    if not (run_dir / "bundle.joblib").exists():
        return None
    bundle = joblib.load(run_dir / "bundle.joblib")
    bundle["models"] = joblib.load(run_dir / "sklearn_models.joblib")
    weights_path = run_dir / "bpn_weights.joblib"
    if weights_path.exists():
        bundle["bpn"] = BPNClassifier.from_weights(joblib.load(weights_path))
    bundle["run_dir"] = str(run_dir)
    return bundle


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
