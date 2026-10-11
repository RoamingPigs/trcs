#!/usr/bin/env python3
"""E1: what do the models write for non-speech clips, with no gate, the energy gate, and a neural VAD gate?

    python run_e1.py --data data/trcs --out results/e1

Hallucination = a non-empty normalised transcript on a non-speech clip. No WER is computed here.
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
    clips = sorted((f for f in man["files"] if f["split"] == "hallucination"), key=lambda f: f["filename"])
    if a.limit:
        clips = clips[: a.limit]
    audio = [data.read_audio(a.data, f) for f in clips]
    # arm -> per clip (active?, audio to transcribe, ratio). 'none' always transcribes the raw clip.
    plans = {"none": [(True, x, 1.0) for x in audio],
             "energy": [gate.gate_decision(x) for x in audio],
             "neural": [gate.neural_gate(x) for x in audio]}
    a.out.mkdir(parents=True, exist_ok=True)
    wanted = [m for m in models.MODELS if a.models == "all" or m["id"] in a.models.split(",")]
    for spec in wanted:
        path = a.out / f"{spec['id']}.jsonl"
        if path.exists():
            print(f"skip {spec['id']}")
            continue
        t0 = time.perf_counter()
        proc, model = models.load(spec)
        rows = []
        for arm, plan in plans.items():
            idx = [i for i, p in enumerate(plan) if p[0]]
            outs = models.transcribe(proc, model, spec, [plan[i][1] for i in idx]) if idx else []
            got = dict(zip(idx, outs))
            for i, f in enumerate(clips):
                active, _, ratio = plan[i]
                o = got.get(i, {"text": "", "tokens": 0, "hit_cap": False})
                words = text.norm_words(o["text"])
                rows.append({
                    "model": spec["id"], "arm": arm, "file": f["filename"], "sha256": f["sha256"],
                    "condition": f["condition"], "seed": int(re.search(r"_s(\d+)\.wav$", f["filename"]).group(1)),
                    "is_speech": f["condition"] == "speech_control", "skipped": not active, "active_ratio": round(float(ratio), 4),
                    "hypothesis": o["text"], "norm_words": len(words), "hallucinated": len(words) > 0,
                    "tokens": o["tokens"], "hit_cap": o["hit_cap"],
                })
        io.write_jsonl(path, rows)
        print(f"{spec['id']}: {len(rows)} rows in {time.perf_counter() - t0:.0f}s")
        del model, proc
    return 0


if __name__ == "__main__":
    sys.exit(main())
