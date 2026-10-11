import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trcs_eval import gate, stats, text  # noqa: E402


def test_edit_counts_known_cases():
    assert text.edit_counts("a b c".split(), "a b c".split()) == (0, 0, 0)
    assert text.edit_counts("a b c".split(), "a x c".split()) == (1, 0, 0)
    assert text.edit_counts("a b c".split(), "a c".split()) == (0, 1, 0)
    assert text.edit_counts("a c".split(), "a b c".split()) == (0, 0, 1)
    assert text.edit_counts([], "a b".split()) == (0, 0, 2)


def test_wer_can_exceed_one_through_insertions():
    s = text.score("cutter munro", "the the the the the cutter munro", normalise=False)
    assert s["edits"] == 5 and s["n"] == 2 and s["edits"] / s["n"] > 1


def test_normaliser_handles_call_signs_and_numbers():
    assert text.normalize("Cutter Waesche CG-751, channel sixteen.") == "cutter waesche cg 751 channel 16"


def test_entity_matching_is_contiguous():
    missed, total = text.entity_misses(["cg-7501", "bertholf"], "this is bertholf cg 7501 standing by")
    assert (missed, total) == (0, 2)
    missed, total = text.entity_misses(["cg-7501"], "cg 75 01")
    assert missed == 1


def test_corpus_wer_is_ratio_of_sums_not_mean_of_ratios():
    e = np.array([[1.0, 9.0]])
    n = np.array([[1.0, 99.0]])
    point, _ = stats.corpus_wer(e, n)
    assert math.isclose(point[0], 10 / 100)


def test_bootstrap_interval_contains_truth_in_a_simulation():
    rng = np.random.default_rng(1)
    cover = 0
    trials = 200
    true = 0.30
    for t in range(trials):
        n = rng.integers(8, 14, size=40)
        e = rng.binomial(n, true).astype(float)
        idx = stats.boot_indices(40, 1000, seed=t)
        _, d = stats.corpus_wer(e[None, :], n.astype(float)[None, :], idx)
        lo, hi = stats.percentile_ci(d[0])
        cover += lo <= true <= hi
    assert 0.88 <= cover / trials <= 1.0, cover / trials


def test_paired_difference_uses_same_resamples():
    e1 = np.array([[2.0, 4.0, 6.0, 8.0]])
    e2 = e1 + 1
    n = np.full((1, 4), 10.0)
    idx = stats.boot_indices(4, 500)
    _, d1 = stats.corpus_wer(e1, n, idx)
    _, d2 = stats.corpus_wer(e2, n, idx)
    assert np.allclose(d2 - d1, 0.1)


def test_wilson_interval_known_values():
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
    lo, hi = stats.wilson(5, 10)
    assert 0.23 < lo < 0.25 and 0.75 < hi < 0.78


def test_energy_gate_skips_digital_silence_and_passes_a_tone():
    sr = 16000
    assert gate.gate_decision(np.zeros(int(2.5 * sr), dtype=np.float32), sr)[0] is False
    t = np.arange(int(2.5 * sr)) / sr
    assert gate.gate_decision((0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)[0] is True
