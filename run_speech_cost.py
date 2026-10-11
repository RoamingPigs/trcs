#!/usr/bin/env python3
"""What does each gate cost on real speech? Score the 80 uncompressed clips (clean and 0 dB) with each gate.

A gate that removes hallucination but also removes speech is not a fix. Arms: none, energy, neural.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".vendor"))
from trcs_eval import data, gate, io, models, text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if data.verify(a.data):
        print("dataset does not match its manifest", file=sys.stderr)
        return 1
    man = data.load_manifest(a.data)
    clips = sorted((f for f in man["files"] if f["split"] == "telephone_grid" and f["codec"] == "none" and f["snr_db"] in (None, 0.0)), key=lambda f: f["filename"])
    if a.limit:
        clips = clips[: a.limit]
    audio = [data.read_audio(a.data, f) for f in clips]
    plans = {"none": [(True, x, 1.0) for x in audio], "energy": [gate.gate_decision(x) for x in audio], "neural": [gate.neural_gate(x) for x in audio]}
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in [m for m in models.MODELS if a.models == "all" or m["id"] in a.models.split(",")]:
        path = a.out / f"{spec['id']}.jsonl"
        if path.exists():
            continue
        t0 = time.perf_counter()
        proc, model = models.load(spec)
        rows = []
        for arm, plan in plans.items():
            idx = [i for i, p in enumerate(plan) if p[0]]
            outs = models.transcribe(proc, model, spec, [plan[i][1] for i in idx]) if idx else []
            got = dict(zip(idx, outs))
            for i, f in enumerate(clips):
                o = got.get(i, {"text": "", "tokens": 0, "hit_cap": False})
                rows.append({"model": spec["id"], "arm": arm, "file": f["filename"], "utterance": "CG_" + re.search(r"e2_cg_(\d+)_", f["filename"]).group(1),
                             "snr_db": f["snr_db"], "skipped": not plan[i][0], "hypothesis": o["text"], "tokens": o["tokens"],
                             "norm": text.score(f["text"], o["text"], True)})
        io.write_jsonl(path, rows)
        print(f"{spec['id']}: {len(rows)} rows in {time.perf_counter() - t0:.0f}s")
        del model, proc
    return 0


if __name__ == "__main__":
    sys.exit(main())
