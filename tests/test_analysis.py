"""Run analyze.py end to end on synthetic per-clip results with a known structure."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trcs_eval import io, models  # noqa: E402

CODECS = ["none", "g711_mulaw", "g711_alaw", "amr_nb", "gsm_fr", "opus_nb"]
NOISE = [None, 20.0, 10.0, 0.0]


def fake(root: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    for spec in models.MODELS:
        rows = []
        for u in range(40):
            n = int(rng.integers(14, 24))
            for c in CODECS:
                for s in NOISE:
                    p = 0.1 + (0.0 if s is None else {20.0: 0.02, 10.0: 0.1, 0.0: 0.4}[s]) + (0.05 if c != "none" else 0)
                    e = int(rng.binomial(n, min(p, 0.95)))
                    sc = {"n": n, "s": e, "d": 0, "i": 0, "edits": e, "hyp_words": n}
                    rows.append({"model": spec["id"], "file": f"e2_cg_{u:02d}_{c}_{s}.wav", "sha256": "x", "utterance": f"CG_{u:02d}", "codec": c, "snr_db": s,
                                 "reference": "r", "hypothesis": "h", "tokens": 20, "hit_cap": False, "norm": sc, "raw": sc, "entities_missed": 1, "entities_total": 5})
        io.write_jsonl(root / "e2" / f"{spec['id']}.jsonl", rows)
        conds = ["digital_silence", "ambient_silence", "noise_white", "speech_control"]
        rows = []
        for arm in ("none", "energy", "neural"):
            for cnd in conds:
                for sd in range(10):
                    sha = f"{cnd}" if cnd == "digital_silence" else f"{cnd}{sd}"
                    h = (arm == "none") or (cnd == "noise_white" and arm == "energy")
                    rows.append({"model": spec["id"], "arm": arm, "file": f"e1_{cnd}_s{sd}.wav", "sha256": sha, "condition": cnd, "seed": sd, "is_speech": cnd == "speech_control",
                                 "skipped": not h, "active_ratio": 0.5, "hypothesis": "you" if h else "", "norm_words": 1 if h else 0, "hallucinated": h, "tokens": 2, "hit_cap": False})
        io.write_jsonl(root / "e1" / f"{spec['id']}.jsonl", rows)
        rows = []
        for arm in ("none", "energy", "neural"):
            for u in range(40):
                for s in (None, 0.0):
                    e = int(rng.binomial(18, 0.1 if s is None else 0.4)) + (3 if arm == "energy" else 0)
                    rows.append({"model": spec["id"], "arm": arm, "file": f"e2_cg_{u:02d}_none_{s}.wav", "utterance": f"CG_{u:02d}", "snr_db": s, "skipped": False, "hypothesis": "h", "tokens": 10,
                                 "norm": {"n": 18, "s": e, "d": 0, "i": 0, "edits": e, "hyp_words": 18}})
        io.write_jsonl(root / "speech_cost" / f"{spec['id']}.jsonl", rows)


def test_analysis_runs_and_has_expected_structure(tmp_path):
    fake(tmp_path)
    out, rep = tmp_path / "analysis.json", tmp_path / "REPORT.md"
    r = subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--results", str(tmp_path), "--out", str(out), "--report", str(rep)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    a = json.loads(out.read_text())
    by = a["e2"]["normalised"]["A1_by_noise"]
    assert by["clean"]["point"] < by["20 dB"]["point"] < by["10 dB"]["point"] < by["0 dB"]["point"]
    for v in by.values():
        assert v["lo"] <= v["point"] <= v["hi"]
    d = a["e2"]["normalised"]["A3_codec_minus_uncompressed_clean"]["amr_nb"]
    assert d["lo"] > 0, "a 0.05 injected codec effect should be detected with 40 utterances x 6 models"
    e1 = a["e1"]["E1_1_rate_by_arm_and_model"]
    assert e1["none|whisper-tiny.en"]["n_distinct_clips"] == 21  # 1 digital silence + 10 + 10
    assert e1["energy|whisper-tiny.en"]["k"] == 10  # only white noise passes the fake energy gate
    assert a["speech_cost"]["delta_energy_minus_none"]["point"] > 0
    assert "E2, word error rate" in rep.read_text()
