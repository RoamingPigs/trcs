"""The energy gate used in experiment E1 (ported unchanged from the pilot).

A frame is active when its RMS is at or above the threshold; a hangover keeps the gate open for a few
frames afterwards. A clip is passed to the model only when at least 15% of frames are active, and
inactive frames are zeroed. This is a level detector. It cannot tell noise from speech.
"""
from __future__ import annotations

import math

import numpy as np


def energy_vad(audio: np.ndarray, sample_rate: int, frame_ms: float = 20.0, hop_ms: float = 10.0,
               threshold_dbfs: float = -42.0, hangover_frames: int = 3) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return {"active_ratio": 0.0, "gated_audio": audio}
    frame_len = max(1, int(round(frame_ms / 1000.0 * sample_rate)))
    hop_len = max(1, int(round(hop_ms / 1000.0 * sample_rate)))
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
    thr = 10.0 ** (threshold_dbfs / 20.0)
    decisions, hang = [], 0
    for i in range(n_frames):
        chunk = audio[i * hop_len : i * hop_len + frame_len]
        rms = math.sqrt(float(np.mean(np.square(chunk.astype(np.float64)))))
        if rms >= thr:
            decisions.append(True)
            hang = hangover_frames
        elif hang > 0:
            decisions.append(True)
            hang -= 1
        else:
            decisions.append(False)
    gated = np.copy(audio)
    for i, dec in enumerate(decisions):
        if not dec:
            gated[i * hop_len : min(i * hop_len + hop_len, len(gated))] = 0.0
    return {"active_ratio": float(sum(decisions) / max(1, len(decisions))), "gated_audio": gated}


ACTIVE_RATIO_MIN = 0.15


def gate_decision(audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, np.ndarray, float]:
    r = energy_vad(audio, sample_rate)
    return r["active_ratio"] >= ACTIVE_RATIO_MIN, r["gated_audio"], r["active_ratio"]


def neural_gate(audio: np.ndarray, sample_rate: int = 16000, _cache: dict = {}) -> tuple[bool, np.ndarray, float]:
    """Neural voice-activity gate (Silero VAD) used as a trigger and a mask, mirroring the energy gate.

    No speech found: the clip is skipped. Otherwise samples outside detected speech are zeroed.
    Returns (active, masked_audio, speech_ratio).
    """
    import torch

    threads = torch.get_num_threads()  # importing the VAD library resets torch to one thread; restore it afterwards
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        if "model" not in _cache:
            _cache["model"] = load_silero_vad(onnx=True)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        ts = get_speech_timestamps(wav, _cache["model"], sampling_rate=sample_rate)
    finally:
        torch.set_num_threads(threads)
    if not ts:
        return False, np.zeros_like(audio), 0.0
    masked = np.zeros_like(audio)
    speech = 0
    for t in ts:
        masked[t["start"] : t["end"]] = audio[t["start"] : t["end"]]
        speech += t["end"] - t["start"]
    return True, masked, speech / max(1, len(audio))


GATES = {"energy": gate_decision, "neural": neural_gate}
