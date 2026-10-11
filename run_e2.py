#!/usr/bin/env python3
"""E2: score every model on every telephone-grid clip of the released dataset.

    python run_e2.py --data data/trcs --out results/e2            # all 960 clips x 6 models
    python run_e2.py --data data/trcs --out /tmp/smoke --limit 12 --models whisper-tiny.en   # smoke test

Resumable: a finished model file is skipped. Greedy decoding, float32, CPU. See PLAN.md.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".vendor"))
from trcs_eval import data, io, models, text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N clips only")
    a = ap.parse_args()

    bad = data.verify(a.data)
    if bad:
        print(f"{len(bad)} files do not match the manifest, e.g. {bad[:3]}", file=sys.stderr)
        return 1
    man = data.load_manifest(a.data)
    clips = sorted((f for f in man["files"] if f["split"] == "telephone_grid"), key=lambda f: f["filename"])
    if a.limit:
        clips = clips[: a.limit]
    audio = [data.read_audio(a.data, f) for f in clips]
    print(f"{len(clips)} clips verified against the manifest")
    a.out.mkdir(parents=True, exist_ok=True)
    io_meta = {"versions": io.versions(), "manifest_generated_at": man["generated_at"], "clips": len(clips),
               "max_new_tokens": models.MAX_NEW_TOKENS, "batch": models.BATCH, "normaliser": [text.NORMALIZER_REPO, text.NORMALIZER_REV]}
    (a.out / "run-meta.json").write_text(__import__("json").dumps(io_meta, indent=1) + "\n", encoding="utf-8")

    wanted = [m for m in models.MODELS if a.models == "all" or m["id"] in a.models.split(",")]
    for spec in wanted:
        path = a.out / f"{spec['id']}.jsonl"
        if path.exists():
            print(f"skip {spec['id']} (done)")
            continue
        t0 = time.perf_counter()
        proc, model = models.load(spec)
        outs = models.transcribe(proc, model, spec, audio)
        rows = []
        for f, o in zip(clips, outs):
            uid = "CG_" + re.search(r"e2_cg_(\d+)_", f["filename"]).group(1)
            hyp = o["text"]
            ref = f["text"]
            miss, tot = text.entity_misses(f.get("entities", []), hyp)
            rows.append({
                "model": spec["id"], "file": f["filename"], "sha256": f["sha256"], "utterance": uid,
                "codec": f["codec"], "snr_db": f["snr_db"], "reference": ref, "hypothesis": hyp,
                "tokens": o["tokens"], "hit_cap": o["hit_cap"],
                "norm": text.score(ref, hyp, True), "raw": text.score(ref, hyp, False),
                "entities_missed": miss, "entities_total": tot,
            })
        # Batch-invariance check: 24 seeded clips decoded one at a time must match the batched text.
        pick = sorted(__import__("numpy").random.default_rng(7).choice(len(clips), min(24, len(clips)), replace=False).tolist())
        single = models.transcribe(proc, model, spec, [audio[i] for i in pick], batch=1)
        same = sum(single[k]["text"] == outs[i]["text"] for k, i in enumerate(pick))
        inv_path = a.out / "batch-invariance.json"
        inv = __import__("json").loads(inv_path.read_text()) if inv_path.exists() else {}
        inv[spec["id"]] = {"checked": len(pick), "identical": same}
        inv_path.write_text(__import__("json").dumps(inv, indent=1) + "\n", encoding="utf-8")
        io.write_jsonl(path, rows)
        print(f"{spec['id']}: {len(rows)} clips in {time.perf_counter() - t0:.0f}s; batch-invariance {same}/{len(pick)}")
        del model, proc
    return 0


if __name__ == "__main__":
    sys.exit(main())
