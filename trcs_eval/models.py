"""The six speech models, pinned to exact Hugging Face revisions, and a batched greedy transcriber."""
from __future__ import annotations

import time

import numpy as np

MODELS = [
    # id, repository, revision, size label (used on the websites), params in millions
    {"id": "whisper-tiny.en", "repo": "openai/whisper-tiny.en", "rev": "87c7102498dcde7456f24cfd30239ca606ed9063", "label": "39M English-only", "m": 39, "multilingual": False},
    {"id": "whisper-base.en", "repo": "openai/whisper-base.en", "rev": "911407f4214e0e1d82085af863093ec0b66f9cd6", "label": "74M English-only", "m": 74, "multilingual": False},
    {"id": "distil-small.en", "repo": "distil-whisper/distil-small.en", "rev": "9e4a67ca4569c30be43a3fe7fba1621e504f0093", "label": "166M English-only (distilled)", "m": 166, "multilingual": False},
    {"id": "whisper-small.en", "repo": "openai/whisper-small.en", "rev": "e8727524f962ee844a7319d92be39ac1bd25655a", "label": "244M English-only", "m": 244, "multilingual": False},
    {"id": "distil-large-v3", "repo": "distil-whisper/distil-large-v3", "rev": "ff05a2254c6f9542535fd778dae85b39a08cae31", "label": "756M multilingual (distilled)", "m": 756, "multilingual": True},
    {"id": "whisper-large-v3-turbo", "repo": "openai/whisper-large-v3-turbo", "rev": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "label": "809M multilingual", "m": 809, "multilingual": True},
]
MAX_NEW_TOKENS = 128
BATCH = 16


def load(spec: dict):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(spec["repo"], revision=spec["rev"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(spec["repo"], revision=spec["rev"], dtype=torch.float32)
    model.eval()
    return proc, model


def transcribe(proc, model, spec: dict, clips: list[np.ndarray], batch: int = BATCH) -> list[dict]:
    """Greedy decoding, no sampling, no beam search. Returns text, generated token count, and whether the cap was hit."""
    import torch

    out = []
    for i in range(0, len(clips), batch):
        chunk = clips[i : i + batch]
        # Default feature extraction pads every clip with zeros to 30 s of audio, so a clip's input never depends on its batch mates.
        feats = proc(chunk, sampling_rate=16000, return_tensors="pt")
        kw = {"max_new_tokens": MAX_NEW_TOKENS, "num_beams": 1, "do_sample": False}
        if spec["multilingual"]:
            kw.update(language="en", task="transcribe")
        t0 = time.perf_counter()
        with torch.inference_mode():
            tokens = model.generate(**feats, **kw)
        dt = time.perf_counter() - t0
        texts = proc.batch_decode(tokens, skip_special_tokens=True)
        pad = proc.tokenizer.pad_token_id
        for t, ids in zip(texts, tokens):
            n = int((ids != pad).sum())
            out.append({"text": t.strip(), "tokens": n, "hit_cap": n >= MAX_NEW_TOKENS, "batch_seconds": round(dt, 3)})
    return out
