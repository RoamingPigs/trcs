#!/usr/bin/env python3
"""Completeness and consistency checks on the raw per-clip results, before any analysis.

    python check_matrix.py --data data/trcs --results results

Fails (exit 1) unless every expected observation is present exactly once, every row has the expected
fields, scores add up, skipped clips have empty output, identical audio gave identical output, and the
run metadata matches the plan. It reads files only. It does not prove the models ran, and it does not
authenticate the machine.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trcs_eval import io, models  # noqa: E402

IDS = [m["id"] for m in models.MODELS]
ARMS = ("none", "energy", "neural")
SCHEMA = {
    "e2": set("model file sha256 utterance codec snr_db reference hypothesis tokens hit_cap norm raw entities_missed entities_total".split()),
    "e1": set("model arm file sha256 condition seed is_speech skipped active_ratio hypothesis norm_words hallucinated tokens hit_cap".split()),
    "speech_cost": set("model arm file utterance snr_db skipped hypothesis tokens norm".split()),
}


def check(data: Path, results: Path) -> list[str]:
    bad: list[str] = []
    def need(ok, msg):
        if not ok:
            bad.append(msg)
    man = json.loads((data / "provenance_manifest.json").read_text(encoding="utf-8"))
    files = man["files"]
    grid = [f for f in files if f["split"] == "telephone_grid"]
    hall = [f for f in files if f["split"] == "hallucination"]
    speech = [f for f in grid if f["codec"] == "none" and f["snr_db"] in (None, 0.0)]
    domains = {"e2": grid, "e1": hall, "speech_cost": speech}
    need((len(grid), len(hall), len(speech)) == (960, 90, 80), f"input dimensions {(len(grid), len(hall), len(speech))}")
    for phase, fs in domains.items():
        names = {p.name for p in (results / phase).glob("*.jsonl")}
        need(names == {m + ".jsonl" for m in IDS}, f"{phase}: model files {sorted(names)}")
        expected = Counter((f["filename"],) if phase == "e2" else (f["filename"], a) for f in fs for a in ((None,) if phase == "e2" else ARMS))
        if phase == "e2":
            expected = Counter((f["filename"],) for f in fs)
        by = {f["filename"]: f for f in fs}
        for mid in IDS:
            p = results / phase / f"{mid}.jsonl"
            if not p.exists():
                continue
            rows = io.read_jsonl(p)
            ident = Counter((r["file"],) if phase == "e2" else (r["file"], r["arm"]) for r in rows)
            need(ident == expected, f"{phase}/{mid}: observation set differs from the manifest ({len(rows)} rows)")
            seen = {}
            for r in rows:
                need(set(r) == SCHEMA[phase], f"{phase}/{mid}: row fields")
                f = by.get(r.get("file"))
                if f is None:
                    continue
                need(r["model"] == mid, f"{phase}/{mid}: model id")
                need(isinstance(r["hypothesis"], str), f"{phase}/{mid}: hypothesis type")
                need(isinstance(r["tokens"], int) and 0 <= r["tokens"] <= models.MAX_NEW_TOKENS, f"{phase}/{mid}: token count")
                if "hit_cap" in r:
                    need(r["hit_cap"] == (r["tokens"] >= models.MAX_NEW_TOKENS), f"{phase}/{mid}: cap flag")
                if "sha256" in r:
                    need(r["sha256"] == f["sha256"], f"{phase}/{mid}: input digest for {r['file']}")
                for key in ("norm", "raw"):
                    if key in r:
                        s = r[key]
                        need(s["edits"] == s["s"] + s["d"] + s["i"] and s["s"] + s["d"] <= s["n"] and s["hyp_words"] == s["n"] - s["d"] + s["i"], f"{phase}/{mid}: score arithmetic")
                if phase == "e1":
                    need(r["hallucinated"] == (r["norm_words"] > 0), f"{phase}/{mid}: hallucination flag")
                    need(r["is_speech"] == (f["condition"] == "speech_control") and r["condition"] == f["condition"], f"{phase}/{mid}: condition flags")
                    need(0 <= r["active_ratio"] <= 1 and math.isfinite(r["active_ratio"]), f"{phase}/{mid}: active ratio")
                    key = (r["arm"], r["sha256"])
                    obs = (r["hypothesis"], r["tokens"], r["skipped"], r["norm_words"])
                    need(key not in seen or seen[key] == obs, f"{phase}/{mid}: identical audio gave different outputs")
                    seen[key] = obs
                else:
                    m = re.search(r"e2_cg_(\d+)_", f["filename"])
                    need(r["utterance"] == "CG_" + m.group(1) and r["snr_db"] == f["snr_db"], f"{phase}/{mid}: utterance or SNR identity")
                    if phase == "e2":
                        need(r["codec"] == f["codec"] and r["reference"] == f["text"], f"{phase}/{mid}: codec or reference")
                if "skipped" in r:
                    need(not (r["arm"] == "none" and r["skipped"]), f"{phase}/{mid}: ungated clip marked skipped")
                    if r["skipped"]:
                        need(r["hypothesis"] == "" and r["tokens"] == 0, f"{phase}/{mid}: skipped clip has output")
    meta = json.loads((results / "e2" / "run-meta.json").read_text())
    need(meta["clips"] == 960 and meta["max_new_tokens"] == 128 and meta["batch"] == 16, "run parameters")
    need(meta["manifest_generated_at"] == man["generated_at"], "manifest date in run metadata")
    v = meta["versions"]
    need(v.get("device") == "cpu" and v.get("dtype") == "float32" and v.get("torch_threads") == 48, "recorded CPU configuration")
    inv = json.loads((results / "e2" / "batch-invariance.json").read_text())
    need(set(inv) == set(IDS), "batch-invariance models")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/trcs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()
    bad = check(a.data, a.results)
    for b in sorted(set(bad))[:40]:
        print("FAIL", b)
    print(f"{len(bad)} problems" if bad else "matrix complete and consistent: 5,760 E2 rows, 1,620 E1 rows, 1,440 speech-cost rows")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
