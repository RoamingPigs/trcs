"""Dataset access: download a pinned revision, verify every file against the manifest, read audio."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ID = "RoamingPigs/trcs-tactical-audio"


def fetch(local_dir: str | Path, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO_ID, repo_type="dataset", revision=revision, local_dir=str(local_dir)))


def load_manifest(root: Path) -> dict:
    return json.loads((Path(root) / "provenance_manifest.json").read_text(encoding="utf-8"))


def verify(root: Path) -> list[str]:
    """Return the names of files whose SHA-256 differs from the manifest (empty list means all match)."""
    root = Path(root)
    bad = []
    for f in load_manifest(root)["files"]:
        if hashlib.sha256((root / f["relative_path"]).read_bytes()).hexdigest() != f["sha256"]:
            bad.append(f["filename"])
    return bad


def read_audio(root: Path, entry: dict) -> np.ndarray:
    data, sr = sf.read(Path(root) / entry["relative_path"], dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{entry['filename']}: expected 16 kHz, got {sr}")
    return data
