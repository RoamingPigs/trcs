#!/usr/bin/env python3
"""Compute every pre-specified analysis in PLAN.md from the raw per-clip results.

    python analyze.py --results results --out results/analysis.json --report results/REPORT.md

Anything not listed in PLAN.md under 'Pre-specified analyses' is not computed here.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from trcs_eval import io, models, stats  # noqa: E402

CODECS = ["none", "g711_mulaw", "g711_alaw", "amr_nb", "gsm_fr", "opus_nb"]
NOISE = [None, 20.0, 10.0, 0.0]
NOISE_LABEL = {None: "clean", 20.0: "20 dB", 10.0: "10 dB", 0.0: "0 dB"}
MIDS = [m["id"] for m in models.MODELS]


def ci(point, draws):
    lo, hi = stats.percentile_ci(draws)
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def load_e2(root: Path):
    utts = None
    cells = {}
    for mi, mid in enumerate(MIDS):
        rows = io.read_jsonl(root / "e2" / f"{mid}.jsonl")
        if utts is None:
            utts = sorted({r["utterance"] for r in rows})
        for r in rows:
            cells[(mi, CODECS.index(r["codec"]), NOISE.index(r["snr_db"]), utts.index(r["utterance"]))] = r
    return utts, cells


def arrays(utts, cells, key):
    shape = (len(MIDS), len(CODECS), len(NOISE), len(utts))
    E, N = np.zeros(shape), np.zeros(shape)
    for (m, c, s, u), r in cells.items():
        E[m, c, s, u] = r[key]["edits"]
        N[m, c, s, u] = r[key]["n"]
    return E, N


def sel(arr, m=None, c=None, s=None):
    def ix(v, n):
        return list(range(n)) if v is None else ([v] if isinstance(v, int) else list(v))
    return arr[np.ix_(ix(m, arr.shape[0]), ix(c, arr.shape[1]), ix(s, arr.shape[2]), range(arr.shape[3]))].sum(axis=(0, 1, 2))


def rate(E, N, idx, **k):
    e, n = sel(E, **k), sel(N, **k)
    return e.sum() / n.sum(), e[idx].sum(1) / n[idx].sum(1)


def diff(E, N, idx, a, b):
    pa, da = rate(E, N, idx, **a)
    pb, db = rate(E, N, idx, **b)
    return pa - pb, da - db


def analyse_e2(root: Path) -> dict:
    utts, cells = load_e2(root)
    idx = stats.boot_indices(len(utts))
    out = {"n_utterances": len(utts), "n_models": len(MIDS), "bootstrap": {"resamples": stats.B_DEFAULT, "seed": stats.SEED, "unit": "utterance", "interval": "percentile 95%"}}
    for name, key in (("normalised", "norm"), ("raw", "raw")):
        E, N = arrays(utts, cells, key)
        block = {}
        block["A1_by_noise"] = {NOISE_LABEL[n]: ci(*rate(E, N, idx, s=si)) for si, n in enumerate(NOISE)}
        block["A2_by_model_clean_uncompressed"] = {MIDS[m]: ci(*rate(E, N, idx, m=m, c=0, s=0)) for m in range(len(MIDS))}
        block["A3_codec_minus_uncompressed_clean"] = {CODECS[c]: ci(*diff(E, N, idx, dict(c=c, s=0), dict(c=0, s=0))) for c in range(1, len(CODECS))}
        block["A4_noise_effect"] = {f"{NOISE_LABEL[n]}_minus_clean": ci(*diff(E, N, idx, dict(s=si), dict(s=0))) for si, n in enumerate(NOISE) if si}
        block["A5_grid"] = {f"{MIDS[m]}|{CODECS[c]}|{NOISE_LABEL[n]}": ci(*rate(E, N, idx, m=m, c=c, s=si)) for m in range(len(MIDS)) for c in range(len(CODECS)) for si, n in enumerate(NOISE)}
        block["by_codec"] = {CODECS[c]: ci(*rate(E, N, idx, c=c)) for c in range(len(CODECS))}
        block["by_model"] = {MIDS[m]: ci(*rate(E, N, idx, m=m)) for m in range(len(MIDS))}
        out[name] = block
    # secondary: share of edits that are insertions, entity misses, cap hits, clips above 100%
    ins = defaultdict(lambda: [0, 0])
    ent = defaultdict(lambda: [0, 0])
    cap = [0, 0]
    over = [0, 0]
    for (m, c, s, u), r in cells.items():
        k = NOISE_LABEL[NOISE[s]]
        ins[k][0] += r["norm"]["i"]
        ins[k][1] += r["norm"]["edits"]
        ent[k][0] += r["entities_missed"]
        ent[k][1] += r["entities_total"]
        cap[0] += int(r["hit_cap"])
        cap[1] += 1
        over[0] += int(r["norm"]["edits"] > r["norm"]["n"])
        over[1] += 1
    out["secondary"] = {
        "insertion_share_of_edits_by_noise": {k: v[0] / v[1] if v[1] else None for k, v in ins.items()},
        "entity_miss_rate_by_noise": {k: v[0] / v[1] for k, v in ent.items()},
        "outputs_hitting_token_cap": {"count": cap[0], "of": cap[1]},
        "clips_with_more_edits_than_reference_words": {"count": over[0], "of": over[1]},
    }
    return out


def analyse_e1(root: Path) -> dict:
    rows = []
    for mid in MIDS:
        rows += io.read_jsonl(root / "e1" / f"{mid}.jsonl")
    nonspeech = [r for r in rows if not r["is_speech"]]
    out = {"unit_note": "Clips with identical audio (same SHA-256) are counted once per model; greedy decoding gives them identical outputs."}
    arms = ["none", "energy", "neural"]
    per_model = {}
    for arm in arms:
        for mid in MIDS:
            seen = {}
            for r in nonspeech:
                if r["arm"] == arm and r["model"] == mid:
                    seen.setdefault(r["sha256"], r)
            k = sum(r["hallucinated"] for r in seen.values())
            n = len(seen)
            lo, hi = stats.wilson(k, n)
            per_model[f"{arm}|{mid}"] = {"k": k, "n_distinct_clips": n, "rate": k / n, "lo": lo, "hi": hi}
    out["E1_1_rate_by_arm_and_model"] = per_model
    conds = sorted({r["condition"] for r in nonspeech})
    by_cond = {}
    for arm in arms:
        for cnd in conds:
            clips = sorted({r["sha256"] for r in nonspeech if r["arm"] == arm and r["condition"] == cnd})
            mat = np.array([[next(x["hallucinated"] for x in nonspeech if x["arm"] == arm and x["model"] == mid and x["sha256"] == h) for h in clips] for mid in MIDS], dtype=float)
            idx = stats.boot_indices(len(clips), 5000)
            draws = mat[:, idx].mean(axis=(0, 2)) if len(clips) > 1 else None
            lo, hi = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) if draws is not None else (None, None)
            by_cond[f"{arm}|{cnd}"] = {"distinct_clips": len(clips), "rate": float(mat.mean()), "lo": lo, "hi": hi}
    out["E1_2_rate_by_arm_and_condition"] = by_cond
    ung = [r for r in nonspeech if r["arm"] == "none"]
    out["E1_3_top_outputs_ungated"] = Counter(" ".join(r["hypothesis"].lower().split()) for r in ung if r["hallucinated"]).most_common(10)
    hall = [r for r in ung if r["hallucinated"]]
    out["E1_4_ungated"] = {"hit_token_cap": sum(r["hit_cap"] for r in hall), "hallucinated_outputs": len(hall), "mean_words_per_hallucinated_output": float(np.mean([r["norm_words"] for r in hall])) if hall else 0.0}
    return out


def analyse_speech_cost(root: Path) -> dict:
    rows = []
    for mid in MIDS:
        rows += io.read_jsonl(root / "speech_cost" / f"{mid}.jsonl")
    utts = sorted({r["utterance"] for r in rows})
    idx = stats.boot_indices(len(utts))
    arms = ["none", "energy", "neural"]
    E = np.zeros((len(MIDS), len(arms), len(utts)))
    N = np.zeros_like(E)
    skipped = np.zeros((len(MIDS), len(arms)))
    for r in rows:
        m, a, u = MIDS.index(r["model"]), arms.index(r["arm"]), utts.index(r["utterance"])
        E[m, a, u] += r["norm"]["edits"]
        N[m, a, u] += r["norm"]["n"]
        skipped[m, a] += int(r["skipped"])
    out = {"clips_per_arm_per_model": 80}
    def cell(a, m=None):
        ms = list(range(len(MIDS))) if m is None else [m]
        e, n = E[ms, a].sum(0), N[ms, a].sum(0)
        return e.sum() / n.sum(), e[idx].sum(1) / n[idx].sum(1)
    for a, arm in enumerate(arms):
        out[f"wer_{arm}"] = ci(*cell(a))
        out[f"clips_skipped_{arm}"] = int(skipped[:, a].sum())
        if a:
            p, d = cell(a)
            p0, d0 = cell(0)
            out[f"delta_{arm}_minus_none"] = ci(p - p0, d - d0)
    return out


def render(res: dict) -> str:
    L = ["# TRCS full-grid results (auto-generated by analyze.py)", "", "Intervals are 95% percentile bootstrap over utterances unless stated. See PLAN.md.", ""]
    e2 = res["e2"]
    for name in ("normalised", "raw"):
        L += [f"## E2, word error rate, {name} text", "", "### By noise level (all models, all audio paths)", "", "| Noise | WER | 95% CI |", "|---|---:|---|"]
        for k, v in e2[name]["A1_by_noise"].items():
            L.append(f"| {k} | {v['point']*100:.1f}% | {v['lo']*100:.1f} to {v['hi']*100:.1f}% |")
        L += ["", "### Clean, uncompressed, by model", "", "| Model | WER | 95% CI |", "|---|---:|---|"]
        for k, v in e2[name]["A2_by_model_clean_uncompressed"].items():
            L.append(f"| {k} | {v['point']*100:.1f}% | {v['lo']*100:.1f} to {v['hi']*100:.1f}% |")
        L += ["", "### Codec minus uncompressed, clean audio (paired, percentage points)", "", "| Codec | Difference | 95% CI | CI excludes 0 |", "|---|---:|---|---|"]
        for k, v in e2[name]["A3_codec_minus_uncompressed_clean"].items():
            L.append(f"| {k} | {v['point']*100:+.1f} | {v['lo']*100:+.1f} to {v['hi']*100:+.1f} | {'yes' if v['lo'] > 0 or v['hi'] < 0 else 'no'} |")
        L += ["", "### Noise minus clean (paired, percentage points)", "", "| Contrast | Difference | 95% CI |", "|---|---:|---|"]
        for k, v in e2[name]["A4_noise_effect"].items():
            L.append(f"| {k} | {v['point']*100:+.1f} | {v['lo']*100:+.1f} to {v['hi']*100:+.1f} |")
        L.append("")
    s = e2["secondary"]
    L += ["## E2 secondary", "", f"- Outputs that hit the {models.MAX_NEW_TOKENS}-token cap: {s['outputs_hitting_token_cap']['count']} of {s['outputs_hitting_token_cap']['of']}",
          f"- Clips with more edits than reference words: {s['clips_with_more_edits_than_reference_words']['count']} of {s['clips_with_more_edits_than_reference_words']['of']}", ""]
    e1 = res["e1"]
    L += ["## E1, words written for non-speech clips", "", "Rate = share of distinct non-speech clips with any word in the normalised output.", "", "| Arm | Model | Rate | 95% Wilson | Distinct clips |", "|---|---|---:|---|---:|"]
    for k, v in e1["E1_1_rate_by_arm_and_model"].items():
        arm, mid = k.split("|")
        L.append(f"| {arm} | {mid} | {v['rate']*100:.0f}% | {v['lo']*100:.0f} to {v['hi']*100:.0f}% | {v['n_distinct_clips']} |")
    L += ["", "### By condition (pooled over models)", "", "| Arm | Condition | Rate | Distinct clips |", "|---|---|---:|---:|"]
    for k, v in e1["E1_2_rate_by_arm_and_condition"].items():
        arm, c = k.split("|")
        L.append(f"| {arm} | {c} | {v['rate']*100:.0f}% | {v['distinct_clips']} |")
    L += ["", "### Most frequent ungated outputs", ""] + [f"- {n}x `{t}`" for t, n in e1["E1_3_top_outputs_ungated"]]
    sc = res["speech_cost"]
    L += ["", "## Cost of each gate on speech (80 uncompressed clips per model)", "", "| Arm | WER | 95% CI | Clips skipped |", "|---|---:|---|---:|"]
    for arm in ("none", "energy", "neural"):
        v = sc[f"wer_{arm}"]
        L.append(f"| {arm} | {v['point']*100:.1f}% | {v['lo']*100:.1f} to {v['hi']*100:.1f}% | {sc[f'clips_skipped_{arm}']} |")
    for arm in ("energy", "neural"):
        v = sc[f"delta_{arm}_minus_none"]
        L.append(f"\n{arm} minus none: {v['point']*100:+.1f} points (95% CI {v['lo']*100:+.1f} to {v['hi']*100:+.1f}).")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis.json"))
    ap.add_argument("--report", type=Path, default=Path("results/REPORT.md"))
    a = ap.parse_args()
    res = {"e2": analyse_e2(a.results), "e1": analyse_e1(a.results), "speech_cost": analyse_speech_cost(a.results)}
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    a.report.write_text(render(res), encoding="utf-8")
    print(f"wrote {a.out} and {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
