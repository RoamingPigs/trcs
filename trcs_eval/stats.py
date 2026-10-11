"""Cluster bootstrap for corpus word error rate, and Wilson intervals for proportions.

The 40 scripted utterances recur in every audio path and noise level, so the clip is not the
independent unit; the utterance is. Resampling utterances (with replacement) and recomputing the
statistic keeps that dependence. Paired differences use the same resampled indices for both arms.
"""
from __future__ import annotations

import math

import numpy as np

B_DEFAULT = 10_000
SEED = 20261011


def boot_indices(n: int, b: int = B_DEFAULT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def corpus_wer(edits: np.ndarray, words: np.ndarray, idx: np.ndarray | None = None):
    """edits, words: arrays whose last axis is the utterance. Returns point estimate and bootstrap draws.

    Statistic = sum(edits) / sum(words) over the utterance axis (a corpus-level rate).
    """
    point = edits.sum(-1) / words.sum(-1)
    if idx is None:
        return point, None
    e = np.take(edits, idx, axis=-1).sum(-1)  # (..., B)
    w = np.take(words, idx, axis=-1).sum(-1)
    return point, e / w


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    return np.quantile(draws, a, axis=-1), np.quantile(draws, 1 - a, axis=-1)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)
