"""Text normalisation and word error counts.

Primary scoring uses the English text normaliser that ships with the Whisper tokenizer
(from a pinned revision), applied identically to the reference and to every model's output.
The pilot's simpler tokenisation (lower-case, \\b[\\w'-]+\\b) is kept as `raw_*` for comparison.
"""
from __future__ import annotations

import re
from functools import lru_cache

NORMALIZER_REPO = "openai/whisper-base.en"
NORMALIZER_REV = "911407f4214e0e1d82085af863093ec0b66f9cd6"


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained(NORMALIZER_REPO, revision=NORMALIZER_REV)


def normalize(text: str) -> str:
    return _tokenizer().normalize(text)


def norm_words(text: str) -> list[str]:
    return normalize(text).split()


def raw_words(text: str) -> list[str]:
    return re.findall(r"\b[\w\-']+\b", text.lower())


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) from a minimum-edit alignment.

    Ties are broken in the order substitution, deletion, insertion so the counts are deterministic.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    s = dl = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dl, ins


def score(reference: str, hypothesis: str, normalise: bool = True) -> dict:
    ref = norm_words(reference) if normalise else raw_words(reference)
    hyp = norm_words(hypothesis) if normalise else raw_words(hypothesis)
    s, d, i = edit_counts(ref, hyp)
    return {"n": len(ref), "s": s, "d": d, "i": i, "edits": s + d + i, "hyp_words": len(hyp)}


def entity_misses(entities: list[str], hypothesis: str) -> tuple[int, int]:
    """Count expected entities missing from the normalised hypothesis.

    An entity matches when its normalised words appear contiguously in the normalised hypothesis.
    Returns (missed, total).
    """
    hyp = norm_words(hypothesis)
    missed = 0
    for e in entities:
        ew = norm_words(e)
        if not ew:
            continue
        found = any(hyp[k : k + len(ew)] == ew for k in range(len(hyp) - len(ew) + 1))
        missed += 0 if found else 1
    return missed, len([e for e in entities if norm_words(e)])
