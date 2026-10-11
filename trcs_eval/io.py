"""Small helpers shared by the runners: environment record, JSONL read/write."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def versions() -> dict:
    import numpy
    import scipy
    import soundfile
    import torch
    import transformers

    try:
        from importlib.metadata import version as _v

        sv = _v("silero_vad")
    except Exception:  # vendored, optional
        sv = None
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[:1]
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
    except Exception:
        cpu = platform.processor()
    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__,
        "silero_vad": sv, "ffmpeg": ff[0] if ff else None, "cpu": cpu, "cores": __import__("os").cpu_count(),
        "torch_threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
