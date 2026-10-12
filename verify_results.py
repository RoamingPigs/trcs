#!/usr/bin/env python3
"""Check published results, or compare your own run with the published one.

    python verify_results.py --published results                    # re-hash files, re-derive the analysis
    python verify_results.py --published results --mine my-results  # compare a reproduction
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trcs_eval import io, models  # noqa: E402

TOL = json.loads((Path(__file__).parent / "expected/tolerances.json").read_text())


def check_hashes(root: Path) -> bool:
    sums = root / "SHA256SUMS"
    if not sums.exists():
        print("no SHA256SUMS in", root)
        return False
    bad = 0
    for line in sums.read_text().splitlines():
        digest, name = line.split(None, 1)
        p = root / name.strip().lstrip("*")
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            print("MISMATCH", name)
            bad += 1
    print(f"{len(sums.read_text().splitlines())} files hashed, {bad} mismatches")
    return bad == 0


def rederive(root: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        out, rep = Path(td) / "a.json", Path(td) / "r.md"
        subprocess.run([sys.executable, str(Path(__file__).parent / "analyze.py"), "--results", str(root), "--out", str(out), "--report", str(rep)], check=True, capture_output=True)
        same = json.loads(out.read_text()) == json.loads((root / "analysis.json").read_text())
    print("analysis.json re-derived from the raw per-clip files:", "identical" if same else "DIFFERENT")
    return same


def compare(pub: Path, mine: Path) -> bool:
    ok = True
    share_min = TOL["same_software_and_hardware_class"]["per_clip_transcripts_identical_min_share"]
    for spec in models.MODELS:
        a = {r["file"]: r["hypothesis"] for r in io.read_jsonl(pub / "e2" / f"{spec['id']}.jsonl")}
        b = {r["file"]: r["hypothesis"] for r in io.read_jsonl(mine / "e2" / f"{spec['id']}.jsonl")}
        share = sum(a[k] == b.get(k) for k in a) / len(a)
        print(f"{spec['id']:26s} identical transcripts: {share:6.1%}  ({'ok' if share >= share_min else 'below ' + format(share_min, '.0%')})")
        ok &= share >= TOL["different_hardware_or_library_version"]["per_clip_transcripts_identical_min_share"]
    A, B = json.loads((pub / "analysis.json").read_text()), json.loads((mine / "analysis.json").read_text())
    for k, v in A["e2"]["normalised"]["A1_by_noise"].items():
        w = B["e2"]["normalised"]["A1_by_noise"][k]
        inside = (w["lo"] <= v["point"] <= w["hi"]) or (v["lo"] <= w["point"] <= v["hi"])
        print(f"noise {k:6s} published {v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]  yours {w['point']:.3f} [{w['lo']:.3f}, {w['hi']:.3f}]  {'agree' if inside else 'DISAGREE'}")
        ok &= inside
    print("RESULT:", "within the stated tolerances" if ok else "outside the stated tolerances; please file a reproduction report either way")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--mine", type=Path)
    ap.add_argument("--data", type=Path, help="also check the result matrix against the dataset manifest (needs the audio manifest)")
    a = ap.parse_args()
    ok = check_hashes(a.published) and rederive(a.published)
    if a.data:
        import check_matrix

        bad = check_matrix.check(a.data, a.published)
        print("matrix check:", "complete and consistent" if not bad else f"{len(set(bad))} problems, e.g. {sorted(set(bad))[:3]}")
        ok = ok and not bad
    if a.mine:
        ok = compare(a.published, a.mine) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
