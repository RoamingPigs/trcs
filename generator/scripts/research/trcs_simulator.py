#!/usr/bin/env python3
"""Tactical Radio Channel Simulator (TRCS) & Acoustic Hallucination Harness.

Simulates acoustic, channel, and codec degradations characteristic of
tactical VHF/UHF, marine, and narrowband telephony communications:
- Silence: digital zero silence, ambient room/circuit noise floor.
- Tactical Squelch: carrier opening click, squelch tail RF noise burst,
  CTCSS sub-audible tone, squelch chatter / hysteresis.
- Non-speech acoustic noise: white, pink, brown, engine drone, babble.
- Narrowband telephone codecs:
    * 8 kHz G.711 mu-law (PCMU)
    * 8 kHz G.711 A-law (PCMA)
    * 8 kHz AMR-NB (Adaptive Multi-Rate NarrowBand, 4.75 to 12.2 kbps)
    * 8 kHz Opus narrowband (VOIP/audio modes)
- Controlled SNR scaling and additive acoustic mixing.
- Acoustic feature extraction: RMS, peak, Crest factor, spectral centroid,
  zero-crossing rate (ZCR), spectral rolloff, sub-band energies.
- Voice Activity Detection (VAD) / Squelch energy gating with hangover.
- Empirical evaluation harness for E1 (hallucination on silence/noise)
  and E2 (telephone-codec grid degradation).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import signal
import scipy.io.wavfile as wavfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional audioop support (Python <=3.12 built-in, or audioop-lts)
try:
    import audioop  # type: ignore
    _HAS_AUDIOOP = True
except ImportError:
    _HAS_AUDIOOP = False


# ==============================================================================
# Mono Audio Coercion & Format Standardization
# ==============================================================================

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure audio array is 1D float32 mono in range [-1.0, 1.0].

    Gracefully handles:
    - 1D mono signals
    - 2D channels-last signals [samples, channels]
    - 2D channels-first signals [channels, samples] (torchaudio / soundfile layout)
    - Multi-channel formats (stereo, 5.1 surround)
    - Integer formats (int16, int32) normalized to [-1.0, 1.0]
    - Empty arrays
    """
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
    elif np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        max_abs = max(abs(info.min), abs(info.max))
        audio = audio.astype(np.float32) / float(max_abs)
    else:
        audio = audio.astype(np.float32)

    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    if audio.ndim == 1:
        out = audio
    elif audio.ndim == 2:
        # Determine channel axis: typical audio has <= 8 channels
        # Shape (C, N) where C <= 8 and N > C -> channels-first, average across axis 0
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            out = np.mean(audio, axis=0)
        else:
            # Shape (N, C) -> channels-last, average across axis 1
            out = np.mean(audio, axis=1)
    else:
        reshaped = audio.reshape(-1, audio.shape[-1])
        out = np.mean(reshaped, axis=-1)

    return np.clip(out.astype(np.float32), -1.0, 1.0)


# ==============================================================================
# OpenCORE AMR-NB C-API ctypes binding (libopencore-amrnb)
# ==============================================================================

class _AmrNbLib:
    """Lazy loader and wrapper for system libopencore-amrnb shared library."""

    _instance: Optional["_AmrNbLib"] = None

    def __init__(self) -> None:
        self.available = False
        lib_name = ctypes.util.find_library("opencore-amrnb") or "libopencore-amrnb.so.0"
        try:
            self.cdll = ctypes.CDLL(lib_name)
            self.cdll.Encoder_Interface_init.restype = ctypes.c_void_p
            self.cdll.Encoder_Interface_init.argtypes = [ctypes.c_int]
            self.cdll.Encoder_Interface_exit.restype = None
            self.cdll.Encoder_Interface_exit.argtypes = [ctypes.c_void_p]
            self.cdll.Encoder_Interface_Encode.restype = ctypes.c_int
            self.cdll.Encoder_Interface_Encode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_short),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
            ]
            self.cdll.Decoder_Interface_init.restype = ctypes.c_void_p
            self.cdll.Decoder_Interface_init.argtypes = []
            self.cdll.Decoder_Interface_exit.restype = None
            self.cdll.Decoder_Interface_exit.argtypes = [ctypes.c_void_p]
            self.cdll.Decoder_Interface_Decode.restype = None
            self.cdll.Decoder_Interface_Decode.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_short),
                ctypes.c_int,
            ]
            self.available = True
        except Exception:
            self.available = False

    @classmethod
    def get(cls) -> "_AmrNbLib":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ==============================================================================
# Signal Generation: Silence, Noise, Squelch, CTCSS
# ==============================================================================

def generate_silence(
    duration_sec: float,
    sample_rate: int = 16000,
    kind: Literal["digital", "ambient"] = "digital",
    ambient_dbfs: float = -60.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate silence signal.

    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sampling rate in Hz.
        kind: 'digital' (pure zeros) or 'ambient' (low-level thermal/room noise).
        ambient_dbfs: Noise floor level in dBFS for ambient silence.
        seed: Random seed for reproducibility.

    Returns:
        float32 1D numpy array in range [-1.0, 1.0].
    """
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    if kind == "digital":
        return np.zeros(n_samples, dtype=np.float32)

    rng = np.random.default_rng(seed)
    rms_linear = 10.0 ** (ambient_dbfs / 20.0)
    noise = rng.normal(0.0, rms_linear, n_samples).astype(np.float32)
    return np.clip(noise, -1.0, 1.0)


def generate_squelch_tail(
    duration_sec: float = 0.10,
    sample_rate: int = 8000,
    burst_type: Literal["white", "bandpass"] = "bandpass",
    amplitude: float = 0.35,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate an RF squelch tail burst.

    When a tactical FM transceiver unkeys, the discriminator releases an
    unmuted noise burst for 50-150 ms before the audio squelch gate shuts.

    Args:
        duration_sec: Squelch tail duration (typically 0.05 to 0.15s).
        sample_rate: Sampling rate in Hz.
        burst_type: 'white' (unfiltered discriminator noise) or 'bandpass'
                    (filtered discriminator hiss).
        amplitude: Peak amplitude target [0.0, 1.0].
        seed: Random seed.

    Returns:
        float32 1D numpy array.
    """
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, n_samples).astype(np.float32)

    if burst_type == "bandpass" and sample_rate > 2000:
        nyq = sample_rate / 2.0
        low = min(800.0 / nyq, 0.4)
        high = min(3600.0 / nyq, 0.95)
        b, a = signal.butter(2, [low, high], btype="bandpass")
        raw = signal.filtfilt(b, a, raw).astype(np.float32)

    attack_len = min(int(0.002 * sample_rate), n_samples // 4)
    decay_len = min(int(0.005 * sample_rate), n_samples // 4)
    env = np.ones(n_samples, dtype=np.float32)
    if attack_len > 0:
        env[:attack_len] = np.linspace(0.0, 1.0, attack_len, dtype=np.float32)
    if decay_len > 0:
        env[-decay_len:] = np.linspace(1.0, 0.0, decay_len, dtype=np.float32)

    tail = raw * env
    peak = np.max(np.abs(tail))
    if peak > 1e-6:
        tail = tail * (amplitude / peak)
    return np.clip(tail, -1.0, 1.0).astype(np.float32)


def generate_squelch_click(
    duration_sec: float = 0.006,
    sample_rate: int = 8000,
    amplitude: float = 0.6,
) -> np.ndarray:
    """Generate squelch opening or closing transient pop/click."""
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=np.float32)
    click = np.sin(2.0 * np.pi * 3.0 * t) * np.exp(-5.0 * t) * amplitude
    return click.astype(np.float32)


def generate_ctcss(
    duration_sec: float,
    sample_rate: int = 8000,
    tone_hz: float = 100.0,
    level_dbc: float = -18.0,
) -> np.ndarray:
    """Generate sub-audible Continuous Tone-Coded Squelch System (CTCSS) tone."""
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    amplitude = 10.0 ** (level_dbc / 20.0)
    return (amplitude * np.sin(2.0 * np.pi * tone_hz * t)).astype(np.float32)


def generate_noise(
    duration_sec: float,
    sample_rate: int = 16000,
    kind: Literal["white", "pink", "brown", "engine", "babble"] = "white",
    amplitude: float = 0.2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate calibrated non-speech acoustic noise."""
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    rng = np.random.default_rng(seed)

    if kind == "white":
        raw = rng.standard_normal(n_samples).astype(np.float32)
    elif kind in ("pink", "brown"):
        white = rng.standard_normal(n_samples).astype(np.float32)
        X = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        freqs[0] = 1.0
        exponent = 0.5 if kind == "pink" else 1.0
        gain = 1.0 / (np.maximum(freqs, 10.0) ** exponent)
        gain[0] = 0.0
        raw = np.fft.irfft(X * gain, n=n_samples).astype(np.float32)
    elif kind == "engine":
        t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
        f0 = 75.0
        raw = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, 9):
            harm_amp = (1.0 / h) * (0.8 + 0.4 * rng.random())
            harm_phase = rng.uniform(0, 2 * np.pi)
            raw += harm_amp * np.sin(2.0 * np.pi * f0 * h * t + harm_phase).astype(np.float32)
        rumble = rng.standard_normal(n_samples).astype(np.float32)
        b, a = signal.butter(2, min(120.0 / (sample_rate / 2.0), 0.8), btype="low")
        rumble = signal.filtfilt(b, a, rumble).astype(np.float32)
        raw += 1.5 * rumble
    elif kind == "babble":
        t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
        raw = np.zeros(n_samples, dtype=np.float32)
        formants = [(500.0, 100.0), (1500.0, 150.0), (2500.0, 200.0), (3200.0, 250.0)]
        nyq = sample_rate / 2.0
        for center_f, bw in formants:
            if center_f >= nyq:
                continue
            low = max((center_f - bw / 2.0) / nyq, 0.01)
            high = min((center_f + bw / 2.0) / nyq, 0.99)
            if high <= low:
                continue
            b, a = signal.butter(2, [low, high], btype="bandpass")
            f_noise = signal.filtfilt(b, a, rng.standard_normal(n_samples)).astype(np.float32)
            mod_f = rng.uniform(3.0, 5.0)
            envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * mod_f * t + rng.uniform(0, 2 * np.pi))) ** 2
            raw += f_noise * envelope.astype(np.float32)
    else:
        raise ValueError(f"Unknown noise kind: {kind}")

    peak = np.max(np.abs(raw))
    if peak > 1e-6:
        raw = raw * (amplitude / peak)
    return np.clip(raw, -1.0, 1.0).astype(np.float32)


# ==============================================================================
# Calibrated SNR Scaling & Mixing
# ==============================================================================

def compute_power(x: np.ndarray) -> float:
    """Compute mean power of a signal: E[x^2]."""
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.square(x, dtype=np.float64)))


def compute_snr_db(signal_arr: np.ndarray, noise_arr: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in decibels."""
    ps = compute_power(signal_arr)
    pn = compute_power(noise_arr)
    if ps <= 1e-12:
        return -float("inf")
    if pn <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(ps / pn))


def mix_at_snr(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    target_snr_db: float,
    headroom_db: float = -1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Mix clean audio and noise audio at an exact target SNR in dB."""
    clean = ensure_mono(clean_audio).astype(np.float64)
    noise = ensure_mono(noise_audio).astype(np.float64)

    if len(noise) < len(clean):
        repeats = int(math.ceil(len(clean) / max(1, len(noise))))
        noise = np.tile(noise, repeats)[: len(clean)]
    elif len(noise) > len(clean):
        noise = noise[: len(clean)]

    ps = compute_power(clean)
    pn = compute_power(noise)

    if ps <= 1e-12:
        target_noise_amp = 10.0 ** (headroom_db / 20.0)
        p_target = target_noise_amp ** 2
        scale = math.sqrt(p_target / max(pn, 1e-12)) if pn > 1e-12 else 0.0
        scaled_noise = noise * scale
        mixed = scaled_noise
        achieved_snr = -float("inf")
    elif pn <= 1e-12:
        mixed = clean
        achieved_snr = float("inf")
        scale = 0.0
    else:
        pn_target = ps / (10.0 ** (target_snr_db / 10.0))
        scale = math.sqrt(pn_target / pn)
        scaled_noise = noise * scale
        mixed = clean + scaled_noise
        achieved_snr = compute_snr_db(clean, scaled_noise)

    peak = np.max(np.abs(mixed)) if len(mixed) > 0 else 0.0
    headroom_linear = 10.0 ** (headroom_db / 20.0)
    if peak > headroom_linear and peak > 1e-6:
        norm_factor = headroom_linear / peak
        mixed = mixed * norm_factor
    else:
        norm_factor = 1.0

    metrics = {
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": float(achieved_snr),
        "noise_scale_factor": float(scale),
        "norm_factor": float(norm_factor),
        "clean_power": float(ps),
        "noise_power": float(pn),
    }
    return np.clip(mixed.astype(np.float32), -1.0, 1.0), metrics


# ==============================================================================
# Audio Resampling & Bandpass Filtering
# ==============================================================================

def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample 1D float32 audio between sample rates using polyphase filtering."""
    audio = ensure_mono(audio)
    if orig_sr == target_sr or len(audio) == 0:
        return audio

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = signal.resample_poly(audio, up, down).astype(np.float32)
    return resampled


def apply_telephone_bandpass(
    audio: np.ndarray,
    sample_rate: int = 8000,
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """Apply standard ITU-T G.712 telephone speech bandpass filter (300 - 3400 Hz)."""
    audio = ensure_mono(audio)
    if len(audio) == 0:
        return audio
    nyq = sample_rate / 2.0
    low = max(low_cutoff / nyq, 0.001)
    high = min(high_cutoff / nyq, 0.999)
    if low >= high:
        return audio
    b, a = signal.butter(order, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, audio).astype(np.float32)
    return filtered


# ==============================================================================
# Narrowband Telephone Codecs: G.711 mu-law, A-law, AMR-NB, Opus
# ==============================================================================

def _g711_mulaw_encode_pure(samples_int16: np.ndarray) -> np.ndarray:
    """Pure numpy implementation of ITU-T G.711 mu-law encoding (16-bit to 8-bit)."""
    s = np.clip(samples_int16, -32768, 32767).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0x00).astype(np.uint8)
    mag = np.where(s < 0, -s, s).astype(np.int32)
    mag = np.clip(mag + 33, 0, 32767)

    exp = np.zeros_like(mag, dtype=np.uint8)
    for i in range(7, 0, -1):
        threshold = 1 << (i + 7)
        mask = (mag >= threshold) & (exp == 0)
        exp = np.where(mask, i, exp)

    mantissa = ((mag >> (exp + 3)) & 0x0F).astype(np.uint8)
    ulaw = ~(sign | (exp << 4) | mantissa) & 0xFF
    return ulaw.astype(np.uint8)


def _g711_mulaw_decode_pure(ulaw_bytes: np.ndarray) -> np.ndarray:
    """Pure numpy implementation of ITU-T G.711 mu-law decoding (8-bit to 16-bit)."""
    u = ~ulaw_bytes.astype(np.uint8) & 0xFF
    sign = (u & 0x80) != 0
    exp = (u >> 4) & 0x07
    mantissa = (u & 0x0F).astype(np.int32)
    mag = ((mantissa << 3) + 33) << exp
    mag = np.maximum(0, mag - 33)
    samples = np.where(sign, -mag, mag).astype(np.int16)
    return samples


def apply_g711_mulaw(
    audio: np.ndarray,
    sample_rate: int = 16000,
    return_sample_rate: int = 16000,
) -> np.ndarray:
    """Simulate 8 kHz G.711 mu-law (PCMU) telephone transmission."""
    audio_8k = resample_audio(audio, sample_rate, 8000)
    filtered_8k = apply_telephone_bandpass(audio_8k, sample_rate=8000)
    pcm16 = np.clip(filtered_8k * 32767.0, -32768.0, 32767.0).astype(np.int16)

    if _HAS_AUDIOOP:
        raw_bytes = pcm16.tobytes()
        ulaw = audioop.lin2ulaw(raw_bytes, 2)
        dec_bytes = audioop.ulaw2lin(ulaw, 2)
        dec_pcm16 = np.frombuffer(dec_bytes, dtype=np.int16)
    else:
        ulaw_arr = _g711_mulaw_encode_pure(pcm16)
        dec_pcm16 = _g711_mulaw_decode_pure(ulaw_arr)

    dec_float = dec_pcm16.astype(np.float32) / 32767.0
    out = resample_audio(dec_float, 8000, return_sample_rate)
    return np.clip(out, -1.0, 1.0)


def apply_g711_alaw(
    audio: np.ndarray,
    sample_rate: int = 16000,
    return_sample_rate: int = 16000,
) -> np.ndarray:
    """Simulate 8 kHz G.711 A-law (PCMA) telephone transmission."""
    audio_8k = resample_audio(audio, sample_rate, 8000)
    filtered_8k = apply_telephone_bandpass(audio_8k, sample_rate=8000)
    pcm16 = np.clip(filtered_8k * 32767.0, -32768.0, 32767.0).astype(np.int16)

    if _HAS_AUDIOOP:
        raw_bytes = pcm16.tobytes()
        alaw = audioop.lin2alaw(raw_bytes, 2)
        dec_bytes = audioop.alaw2lin(alaw, 2)
        dec_pcm16 = np.frombuffer(dec_bytes, dtype=np.int16)
    else:
        sign = np.sign(pcm16)
        mag = np.abs(pcm16).astype(np.float32) / 32768.0
        A = 87.6
        companded = np.where(
            mag < (1.0 / A),
            (A * mag) / (1.0 + math.log(A)),
            (1.0 + np.log(np.maximum(A * mag, 1e-6))) / (1.0 + math.log(A)),
        )
        quantized = np.round(companded * 127.0) / 127.0
        expanded = np.where(
            quantized < (1.0 / (1.0 + math.log(A))),
            (quantized * (1.0 + math.log(A))) / A,
            np.exp(quantized * (1.0 + math.log(A)) - 1.0) / A,
        )
        dec_pcm16 = np.clip(sign * expanded * 32767.0, -32768.0, 32767.0).astype(np.int16)

    dec_float = dec_pcm16.astype(np.float32) / 32767.0
    out = resample_audio(dec_float, 8000, return_sample_rate)
    return np.clip(out, -1.0, 1.0)


def apply_amr_nb(
    audio: np.ndarray,
    sample_rate: int = 16000,
    return_sample_rate: int = 16000,
    mode: int = 7,  # Mode 7 = MR122 (12.2 kbps)
) -> np.ndarray:
    """Simulate 8 kHz AMR-NB (Adaptive Multi-Rate NarrowBand) transmission."""
    audio_8k = resample_audio(audio, sample_rate, 8000)
    filtered_8k = apply_telephone_bandpass(audio_8k, sample_rate=8000)
    pcm16 = np.clip(filtered_8k * 32767.0, -32768.0, 32767.0).astype(np.int16)

    amr_lib = _AmrNbLib.get()
    if amr_lib.available:
        n_orig = len(pcm16)
        rem = n_orig % 160
        if rem != 0:
            pad_len = 160 - rem
            pcm_padded = np.pad(pcm16, (0, pad_len))
        else:
            pcm_padded = pcm16

        n_frames = len(pcm_padded) // 160
        enc = amr_lib.cdll.Encoder_Interface_init(0)
        dec = amr_lib.cdll.Decoder_Interface_init()

        out_samples = np.zeros_like(pcm_padded)
        out_buf = (ctypes.c_uint8 * 64)()
        dec_buf = (ctypes.c_short * 160)()

        try:
            for i in range(n_frames):
                frame = np.ascontiguousarray(pcm_padded[i * 160 : (i + 1) * 160], dtype=np.int16)
                in_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
                _ = amr_lib.cdll.Encoder_Interface_Encode(enc, mode, in_ptr, out_buf, 0)
                amr_lib.cdll.Decoder_Interface_Decode(dec, out_buf, dec_buf, 0)
                out_samples[i * 160 : (i + 1) * 160] = np.ctypeslib.as_array(dec_buf)
        finally:
            amr_lib.cdll.Encoder_Interface_exit(enc)
            amr_lib.cdll.Decoder_Interface_exit(dec)

        dec_pcm16 = out_samples[:n_orig]
    else:
        dec_pcm16 = _emulate_celp_narrowband(pcm16)

    dec_float = dec_pcm16.astype(np.float32) / 32767.0
    out = resample_audio(dec_float, 8000, return_sample_rate)
    return np.clip(out, -1.0, 1.0)


def _emulate_celp_narrowband(pcm16: np.ndarray) -> np.ndarray:
    """Emulate cellular algebraic code-excited linear prediction (CELP) artifacts."""
    if len(pcm16) < 160:
        return pcm16
    x = pcm16.astype(np.float32) / 32768.0
    pre = signal.lfilter([1.0, -0.68], [1.0], x)
    step = 0.05
    quant = np.round(pre / step) * step
    de_emp = signal.lfilter([1.0], [1.0, -0.68], quant)
    return np.clip(de_emp * 32767.0, -32768.0, 32767.0).astype(np.int16)


def apply_opus_narrowband(
    audio: np.ndarray,
    sample_rate: int = 16000,
    return_sample_rate: int = 16000,
    bitrate: int = 12000,
) -> np.ndarray:
    """Simulate 8 kHz Opus Narrowband transmission (VOIP / mesh radio)."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        try:
            with tempfile.TemporaryDirectory() as td:
                in_wav = os.path.join(td, "in.wav")
                out_ogg = os.path.join(td, "out.opus")
                out_wav = os.path.join(td, "out.wav")

                audio_8k = resample_audio(audio, sample_rate, 8000)
                filtered_8k = apply_telephone_bandpass(audio_8k, sample_rate=8000)
                pcm16 = np.clip(filtered_8k * 32767.0, -32768.0, 32767.0).astype(np.int16)
                wavfile.write(in_wav, 8000, pcm16)

                cmd_enc = [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    in_wav,
                    "-ar",
                    "8000",
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    f"{bitrate}",
                    "-application",
                    "voip",
                    out_ogg,
                ]
                res_enc = subprocess.run(cmd_enc, capture_output=True)
                if res_enc.returncode == 0 and os.path.exists(out_ogg):
                    cmd_dec = [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        out_ogg,
                        "-ar",
                        str(return_sample_rate),
                        "-ac",
                        "1",
                        out_wav,
                    ]
                    res_dec = subprocess.run(cmd_dec, capture_output=True)
                    if res_dec.returncode == 0 and os.path.exists(out_wav):
                        sr_read, data_read = wavfile.read(out_wav)
                        dec_float = ensure_mono(data_read)
                        expected_len = int(round(len(audio) * return_sample_rate / sample_rate))
                        if len(dec_float) < expected_len:
                            dec_float = np.pad(dec_float, (0, expected_len - len(dec_float)))
                        elif len(dec_float) > expected_len:
                            dec_float = dec_float[:expected_len]
                        return np.clip(dec_float, -1.0, 1.0)
        except Exception:
            pass

    audio_8k = resample_audio(audio, sample_rate, 8000)
    filtered = apply_telephone_bandpass(audio_8k, sample_rate=8000, low_cutoff=250.0, high_cutoff=3600.0)
    out = resample_audio(filtered, 8000, return_sample_rate)
    return np.clip(out, -1.0, 1.0)


def apply_gsm_fr(
    audio: np.ndarray,
    sample_rate: int = 16000,
    return_sample_rate: int = 16000,
) -> np.ndarray:
    """Simulate 8 kHz GSM 06.10 Full-Rate (13 kbps) telephone transmission."""
    audio = ensure_mono(audio)
    if len(audio) == 0:
        return audio

    expected_len = int(round(len(audio) * return_sample_rate / sample_rate))

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        try:
            with tempfile.TemporaryDirectory() as td:
                in_wav = os.path.join(td, "in.wav")
                out_gsm = os.path.join(td, "out.gsm")
                out_wav = os.path.join(td, "out.wav")

                audio_8k = resample_audio(audio, sample_rate, 8000)
                filtered_8k = apply_telephone_bandpass(audio_8k, sample_rate=8000)
                pcm16 = np.clip(filtered_8k * 32767.0, -32768.0, 32767.0).astype(np.int16)
                wavfile.write(in_wav, 8000, pcm16)

                cmd_enc = [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    in_wav,
                    "-c:a",
                    "libgsm",
                    "-ar",
                    "8000",
                    "-ac",
                    "1",
                    out_gsm,
                ]
                res_enc = subprocess.run(cmd_enc, capture_output=True)
                if res_enc.returncode == 0 and os.path.exists(out_gsm):
                    cmd_dec = [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        out_gsm,
                        "-ar",
                        str(return_sample_rate),
                        "-ac",
                        "1",
                        out_wav,
                    ]
                    res_dec = subprocess.run(cmd_dec, capture_output=True)
                    if res_dec.returncode == 0 and os.path.exists(out_wav):
                        sr_read, data_read = wavfile.read(out_wav)
                        dec_float = ensure_mono(data_read)
                        if len(dec_float) < expected_len:
                            dec_float = np.pad(dec_float, (0, expected_len - len(dec_float)))
                        elif len(dec_float) > expected_len:
                            dec_float = dec_float[:expected_len]
                        return np.clip(dec_float, -1.0, 1.0)
        except Exception:
            pass

    # Fallback to cellular LPC emulation if ffmpeg / libgsm unavailable
    audio_8k = resample_audio(audio, sample_rate, 8000)
    filtered_8k = apply_telephone_bandpass(audio_8k, sample_rate=8000)
    pcm16 = np.clip(filtered_8k * 32767.0, -32768.0, 32767.0).astype(np.int16)
    dec_pcm16 = _emulate_celp_narrowband(pcm16)
    dec_float = dec_pcm16.astype(np.float32) / 32767.0
    out = resample_audio(dec_float, 8000, return_sample_rate)
    if len(out) < expected_len:
        out = np.pad(out, (0, expected_len - len(out)))
    elif len(out) > expected_len:
        out = out[:expected_len]
    return np.clip(out, -1.0, 1.0)


def apply_codec(
    audio: np.ndarray,
    sample_rate: int,
    codec: Literal["none", "pcm8k", "g711_mulaw", "g711_alaw", "amr_nb", "gsm_fr", "opus_nb"],
    return_sample_rate: int = 16000,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch codec processing to the appropriate engine."""
    audio = ensure_mono(audio)
    if codec == "none":
        return resample_audio(audio, sample_rate, return_sample_rate)
    elif codec == "pcm8k":
        down = resample_audio(audio, sample_rate, 8000)
        filt = apply_telephone_bandpass(down, sample_rate=8000)
        return resample_audio(filt, 8000, return_sample_rate)
    elif codec == "g711_mulaw":
        return apply_g711_mulaw(audio, sample_rate, return_sample_rate)
    elif codec == "g711_alaw":
        return apply_g711_alaw(audio, sample_rate, return_sample_rate)
    elif codec == "amr_nb":
        mode = kwargs.get("mode", 7)
        return apply_amr_nb(audio, sample_rate, return_sample_rate, mode=mode)
    elif codec == "gsm_fr":
        return apply_gsm_fr(audio, sample_rate, return_sample_rate)
    elif codec == "opus_nb":
        bitrate = kwargs.get("bitrate", 12000)
        return apply_opus_narrowband(audio, sample_rate, return_sample_rate, bitrate=bitrate)
    else:
        raise ValueError(f"Unknown codec: {codec}")


# ==============================================================================
# Full Tactical Radio Channel Simulation
# ==============================================================================

@dataclass
class TacticalChannelConfig:
    """Configuration for tactical radio channel simulator."""

    profile_name: str = "tactical_vhf"
    codec: Literal["none", "pcm8k", "g711_mulaw", "g711_alaw", "amr_nb", "gsm_fr", "opus_nb"] = "g711_mulaw"
    snr_db: Optional[float] = 15.0
    noise_kind: Literal["white", "pink", "brown", "engine", "babble"] = "pink"
    add_squelch_click: bool = True
    add_squelch_tail: bool = True
    squelch_tail_sec: float = 0.08
    add_ctcss: bool = True
    ctcss_hz: float = 100.0
    ctcss_level_dbc: float = -18.0
    headroom_db: float = -1.0
    output_sample_rate: int = 16000
    seed: Optional[int] = None


def apply_tactical_channel(
    clean_audio: np.ndarray,
    sample_rate: int,
    config: TacticalChannelConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply complete tactical radio channel degradations to an input signal."""
    clean_audio = ensure_mono(clean_audio)
    rng = np.random.default_rng(config.seed)
    channel_sr = 8000

    degraded_8k = apply_codec(
        clean_audio,
        sample_rate=sample_rate,
        codec=config.codec,
        return_sample_rate=channel_sr,
    )

    if config.add_ctcss and len(degraded_8k) > 0:
        dur = len(degraded_8k) / float(channel_sr)
        ctcss = generate_ctcss(dur, channel_sr, tone_hz=config.ctcss_hz, level_dbc=config.ctcss_level_dbc)
        degraded_8k = degraded_8k + ctcss[: len(degraded_8k)]

    noise_metrics = {}
    if config.snr_db is not None and len(degraded_8k) > 0:
        dur = len(degraded_8k) / float(channel_sr)
        noise = generate_noise(
            dur,
            sample_rate=channel_sr,
            kind=config.noise_kind,
            seed=int(rng.integers(0, 1000000)),
        )
        degraded_8k, noise_metrics = mix_at_snr(
            degraded_8k,
            noise,
            target_snr_db=config.snr_db,
            headroom_db=config.headroom_db,
        )

    tail_arr = np.array([], dtype=np.float32)
    if config.add_squelch_tail:
        tail_arr = generate_squelch_tail(
            duration_sec=config.squelch_tail_sec,
            sample_rate=channel_sr,
            burst_type="bandpass",
            amplitude=0.35,
            seed=int(rng.integers(0, 1000000)),
        )

    click_arr = np.array([], dtype=np.float32)
    if config.add_squelch_click:
        click_arr = generate_squelch_click(
            duration_sec=0.005,
            sample_rate=channel_sr,
            amplitude=0.5,
        )

    assembled_8k = np.concatenate([click_arr, degraded_8k, tail_arr]).astype(np.float32)
    out_audio = resample_audio(assembled_8k, channel_sr, config.output_sample_rate)
    out_audio = np.clip(out_audio, -1.0, 1.0)

    audit_info = {
        "profile_name": config.profile_name,
        "codec": config.codec,
        "channel_sample_rate": channel_sr,
        "output_sample_rate": config.output_sample_rate,
        "input_duration_sec": float(len(clean_audio) / float(sample_rate)),
        "output_duration_sec": float(len(out_audio) / float(config.output_sample_rate)),
        "noise_metrics": noise_metrics,
    }
    return out_audio, audit_info


# ==============================================================================
# Acoustic Feature Extraction & VAD Squelch Gate
# ==============================================================================

def extract_acoustic_features(
    audio: np.ndarray,
    sample_rate: int,
) -> Dict[str, float]:
    """Extract standard acoustic, spectral, and energy features from an audio clip."""
    audio = ensure_mono(audio)
    if len(audio) == 0:
        return {
            "duration_sec": 0.0,
            "peak_amplitude": 0.0,
            "peak_dbfs": -120.0,
            "rms_energy": 0.0,
            "rms_dbfs": -120.0,
            "crest_factor": 1.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "subband_low_ratio": 0.0,
            "subband_telephony_ratio": 0.0,
            "subband_high_ratio": 0.0,
        }

    x = audio.astype(np.float64)
    duration_sec = len(x) / float(sample_rate)

    peak = float(np.max(np.abs(x)))
    peak_dbfs = float(20.0 * math.log10(max(peak, 1e-6)))
    rms = float(math.sqrt(np.mean(np.square(x))))
    rms_dbfs = float(20.0 * math.log10(max(rms, 1e-6)))
    crest = float(peak / max(rms, 1e-6))

    zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(x)))))
    zcr = float(zero_crossings / max(duration_sec, 1e-6))

    n_fft = min(len(x), 4096)
    window = signal.windows.hann(n_fft)
    segment = x[:n_fft] * window
    mag = np.abs(np.fft.rfft(segment))
    power = np.square(mag)
    total_power = np.sum(power)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    if total_power > 1e-12:
        centroid = float(np.sum(freqs * power) / total_power)
        cum_power = np.cumsum(power)
        rolloff_idx = np.searchsorted(cum_power, 0.85 * total_power)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        rolloff = float(freqs[rolloff_idx])
        low_p = float(np.sum(power[freqs < 300.0]))
        tel_p = float(np.sum(power[(freqs >= 300.0) & (freqs <= 3400.0)]))
        high_p = float(np.sum(power[freqs > 3400.0]))
        sub_low = low_p / total_power
        sub_tel = tel_p / total_power
        sub_high = high_p / total_power
    else:
        centroid = 0.0
        rolloff = 0.0
        sub_low = 0.0
        sub_tel = 0.0
        sub_high = 0.0

    return {
        "duration_sec": float(duration_sec),
        "peak_amplitude": float(peak),
        "peak_dbfs": float(peak_dbfs),
        "rms_energy": float(rms),
        "rms_dbfs": float(rms_dbfs),
        "crest_factor": float(crest),
        "zero_crossing_rate": float(zcr),
        "spectral_centroid_hz": float(centroid),
        "spectral_rolloff_hz": float(rolloff),
        "subband_low_ratio": float(sub_low),
        "subband_telephony_ratio": float(sub_tel),
        "subband_high_ratio": float(sub_high),
    }


def energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_dbfs: float = -45.0,
    hangover_frames: int = 6,
) -> Dict[str, Any]:
    """Voice Activity Detection (VAD) / Squelch Energy Gate with hysteresis."""
    audio = ensure_mono(audio)
    if len(audio) == 0:
        return {
            "active_ratio": 0.0,
            "frame_decisions": [],
            "frame_energy_dbfs": [],
            "gated_audio": audio,
        }

    frame_len = max(1, int(round((frame_ms / 1000.0) * sample_rate)))
    hop_len = max(1, int(round((hop_ms / 1000.0) * sample_rate)))
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    decisions = []
    energies = []
    hangover_cnt = 0

    threshold_linear = 10.0 ** (threshold_dbfs / 20.0)

    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        chunk = audio[start:end]
        rms = math.sqrt(np.mean(np.square(chunk.astype(np.float64))))
        e_dbfs = 20.0 * math.log10(max(rms, 1e-6))
        energies.append(e_dbfs)

        if rms >= threshold_linear:
            decisions.append(True)
            hangover_cnt = hangover_frames
        elif hangover_cnt > 0:
            decisions.append(True)
            hangover_cnt -= 1
        else:
            decisions.append(False)

    active_ratio = float(sum(decisions) / max(1, len(decisions)))

    gated = np.copy(audio)
    for i, dec in enumerate(decisions):
        if not dec:
            start = i * hop_len
            end = min(start + hop_len, len(gated))
            gated[start:end] = 0.0

    return {
        "active_ratio": float(active_ratio),
        "frame_decisions": decisions,
        "frame_energy_dbfs": energies,
        "gated_audio": gated,
    }


# ==============================================================================
# Empirical Evaluation Metrics: E1 & E2 Hallucination & Codec Grid
# ==============================================================================

def compute_hallucination_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute ASR Hallucination audit metrics for E1."""
    total_non_speech_clips = 0
    hallucinated_clips = 0
    total_non_speech_duration_sec = 0.0
    total_inserted_words = 0

    by_condition: Dict[str, Dict[str, Any]] = {}

    for r in records:
        is_speech = bool(r.get("is_speech", False))
        if is_speech:
            continue

        raw_transcript = r.get("transcript")
        transcript = str(raw_transcript).strip() if raw_transcript is not None else ""
        words = transcript.split() if transcript else []
        dur = float(r.get("duration_sec", 0.0))
        cond = str(r.get("condition", "unspecified"))

        total_non_speech_clips += 1
        total_non_speech_duration_sec += dur
        total_inserted_words += len(words)
        if len(words) > 0:
            hallucinated_clips += 1

        if cond not in by_condition:
            by_condition[cond] = {
                "clips": 0,
                "hallucinated_clips": 0,
                "duration_sec": 0.0,
                "words": 0,
            }
        by_condition[cond]["clips"] += 1
        by_condition[cond]["duration_sec"] += dur
        by_condition[cond]["words"] += len(words)
        if len(words) > 0:
            by_condition[cond]["hallucinated_clips"] += 1

    total_min = max(total_non_speech_duration_sec / 60.0, 1e-6)
    wpm = total_inserted_words / total_min
    clip_rate = (
        float(hallucinated_clips / total_non_speech_clips)
        if total_non_speech_clips > 0
        else 0.0
    )

    cond_summary = {}
    for c, data in by_condition.items():
        c_min = max(data["duration_sec"] / 60.0, 1e-6)
        cond_summary[c] = {
            "clips": data["clips"],
            "hallucinated_clips": data["hallucinated_clips"],
            "hallucination_clip_rate": float(data["hallucinated_clips"] / max(1, data["clips"])),
            "words_per_minute": float(data["words"] / c_min),
            "total_words": data["words"],
        }

    return {
        "total_non_speech_clips": total_non_speech_clips,
        "hallucinated_clips": hallucinated_clips,
        "hallucination_clip_rate": float(clip_rate),
        "total_non_speech_duration_sec": float(total_non_speech_duration_sec),
        "total_inserted_words": total_inserted_words,
        "words_per_minute_hallucinated": float(wpm),
        "by_condition": cond_summary,
    }


def compute_wer(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Compute Word Error Rate (WER) using Levenshtein distance dynamic programming.

    WER = (Substitutions + Deletions + Insertions) / N_reference

    Args:
        reference: Reference transcript string.
        hypothesis: Recognized hypothesis string.
        case_sensitive: If False (default), comparisons are lowercase.

    Returns:
        Dictionary containing wer, substitutions, deletions, insertions, hits, ref_words, hyp_words.
    """
    def tokenize(s: str) -> List[str]:
        if not case_sensitive:
            s = s.lower()
        return re.findall(r"\b[\w\-']+\b", s)

    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        wer = float(m) if m > 0 else 0.0
        return {
            "wer": float(wer),
            "substitutions": 0,
            "deletions": 0,
            "insertions": m,
            "hits": 0,
            "ref_words": 0,
            "hyp_words": m,
        }

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # Deletion
                    dp[i][j - 1],       # Insertion
                    dp[i - 1][j - 1],   # Substitution
                )

    # Backtrace to count S, D, I, H
    i, j = n, m
    s, d, ins, h = 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            h += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                s += 1
                i -= 1
                j -= 1
            elif i > 0:
                d += 1
                i -= 1
            else:
                ins += 1
                j -= 1

    wer = (s + d + ins) / float(n)
    return {
        "wer": float(wer),
        "substitutions": int(s),
        "deletions": int(d),
        "insertions": int(ins),
        "hits": int(h),
        "ref_words": int(n),
        "hyp_words": int(m),
    }


def extract_tactical_entities(
    text: str,
    custom_callsigns: Optional[List[str]] = None,
) -> List[str]:
    """Extract tactical callsigns and numbers from transcript text."""
    nato_phonetics = {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliett", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
        "xray", "x-ray", "yankee", "zulu",
    }
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
    }
    custom_set = {c.lower() for c in (custom_callsigns or [])}

    # Normalize internal number commas: e.g. "7,501" -> "7501"
    text_clean = re.sub(r"(\d),(\d)", r"\1\2", text.lower())

    # Normalize common naval/tactical callsign variations:
    # "c.g. 7501", "cg 7501", "cg-7501", "uscgc 750", "uscgc-750"
    def _norm_callsign(m: re.Match) -> str:
        prefix = re.sub(r"[\s\.\-]+", "", m.group(1))
        return f" {prefix}-{m.group(2)} "

    text_clean = re.sub(r"\b(c[\s\.]*g|uscgc|nv)[\s\.\-]*(\d{1,6})\b", _norm_callsign, text_clean)
    text_clean = text_clean.replace("/", " ").replace(".", " ")
    tokens = re.findall(r"\b[\w\-]+\b", text_clean)

    entities = []
    for token in tokens:
        if token.isdigit():
            entities.append(token)
        elif token in number_words:
            entities.append(token)
        elif token in nato_phonetics:
            entities.append(token)
        elif re.match(r"^[a-z]{1,6}-\d{1,6}$", token):
            entities.append(token)
        elif re.match(r"^[a-z]{1,6}\d{1,6}$", token) and not token.isdigit():
            # Handle unhyphenated callsigns e.g. "cg7501" -> "cg-7501", "uscgc750" -> "uscgc-750"
            m = re.match(r"^([a-z]{1,6})(\d{1,6})$", token)
            if m:
                entities.append(f"{m.group(1)}-{m.group(2)}")
            else:
                entities.append(token)
        elif token in custom_set:
            entities.append(token)

    return entities


def compute_entity_error_rate(
    reference: str,
    hypothesis: str,
    custom_callsigns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute entity error rate (EER) on tactical callsigns and numbers (E2 metric).

    EER = (False Positives + False Negatives) / max(1, Total Reference Entities)
    """
    ref_entities = extract_tactical_entities(reference, custom_callsigns)
    hyp_entities = extract_tactical_entities(hypothesis, custom_callsigns)

    ref_counts = Counter(ref_entities)
    hyp_counts = Counter(hyp_entities)

    tp = sum((ref_counts & hyp_counts).values())
    fn = len(ref_entities) - tp
    fp = len(hyp_entities) - tp

    n_ref = len(ref_entities)
    n_hyp = len(hyp_entities)

    precision = float(tp / n_hyp) if n_hyp > 0 else (1.0 if n_ref == 0 else 0.0)
    recall = float(tp / n_ref) if n_ref > 0 else (1.0 if n_hyp == 0 else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    eer = float((fn + fp) / max(1, n_ref)) if n_ref > 0 else (1.0 if n_hyp > 0 else 0.0)

    return {
        "entity_error_rate": float(eer),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "reference_entities": ref_entities,
        "hypothesis_entities": hyp_entities,
    }


def evaluate_telephone_codec_grid(
    clean_audio: np.ndarray,
    sample_rate: int,
    codecs: Optional[List[str]] = None,
    snr_levels_db: Optional[List[Optional[float]]] = None,
    noise_kind: Literal["white", "pink", "brown", "engine", "babble"] = "pink",
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Generate telephone-codec grid matrix for E2 evaluation.

    Evaluates clean utterance through combinations of narrowband telephone codecs
    and SNR noise levels, extracting acoustic features for each condition.
    """
    clean = ensure_mono(clean_audio)
    if codecs is None:
        codecs = ["none", "pcm8k", "g711_mulaw", "g711_alaw", "amr_nb", "gsm_fr", "opus_nb"]
    if snr_levels_db is None:
        snr_levels_db = [None, 20.0, 10.0, 0.0]

    grid_results = []
    rng = np.random.default_rng(seed)

    for codec in codecs:
        for snr in snr_levels_db:
            degraded = apply_codec(clean, sample_rate, codec=codec, return_sample_rate=16000)  # type: ignore[arg-type]
            noise_metrics: Dict[str, float] = {}
            if snr is not None and len(degraded) > 0:
                dur = len(degraded) / 16000.0
                noise = generate_noise(dur, sample_rate=16000, kind=noise_kind, seed=int(rng.integers(0, 1000000)))
                degraded, noise_metrics = mix_at_snr(degraded, noise, target_snr_db=snr)

            feats = extract_acoustic_features(degraded, 16000)
            vad = energy_vad(degraded, 16000)

            grid_results.append({
                "codec": codec,
                "target_snr_db": snr,
                "noise_kind": noise_kind if snr is not None else None,
                "noise_metrics": noise_metrics,
                "acoustic_features": feats,
                "vad_active_ratio": vad["active_ratio"],
                "audio_samples": len(degraded),
            })

    return grid_results


# ==============================================================================
# Pre-configured Tactical Profiles
# ==============================================================================

TACTICAL_PROFILES: Dict[str, TacticalChannelConfig] = {
    "clean_control": TacticalChannelConfig(
        profile_name="clean_control",
        codec="none",
        snr_db=None,
        add_squelch_click=False,
        add_squelch_tail=False,
        add_ctcss=False,
    ),
    "pstn_g711": TacticalChannelConfig(
        profile_name="pstn_g711",
        codec="g711_mulaw",
        snr_db=30.0,
        noise_kind="white",
        add_squelch_click=False,
        add_squelch_tail=False,
        add_ctcss=False,
    ),
    "tactical_vhf_clear": TacticalChannelConfig(
        profile_name="tactical_vhf_clear",
        codec="g711_mulaw",
        snr_db=20.0,
        noise_kind="pink",
        add_squelch_click=True,
        add_squelch_tail=True,
        squelch_tail_sec=0.08,
        add_ctcss=True,
        ctcss_hz=100.0,
    ),
    "tactical_vhf_fringe": TacticalChannelConfig(
        profile_name="tactical_vhf_fringe",
        codec="amr_nb",
        snr_db=5.0,
        noise_kind="pink",
        add_squelch_click=True,
        add_squelch_tail=True,
        squelch_tail_sec=0.12,
        add_ctcss=True,
        ctcss_hz=131.8,
    ),
    "tactical_gsm": TacticalChannelConfig(
        profile_name="tactical_gsm",
        codec="gsm_fr",
        snr_db=15.0,
        noise_kind="pink",
        add_squelch_click=True,
        add_squelch_tail=True,
        squelch_tail_sec=0.06,
        add_ctcss=False,
    ),
    "mesh_radio_opus": TacticalChannelConfig(
        profile_name="mesh_radio_opus",
        codec="opus_nb",
        snr_db=15.0,
        noise_kind="babble",
        add_squelch_click=False,
        add_squelch_tail=False,
        add_ctcss=False,
    ),
    "extreme_combat_engine": TacticalChannelConfig(
        profile_name="extreme_combat_engine",
        codec="amr_nb",
        snr_db=-2.0,
        noise_kind="engine",
        add_squelch_click=True,
        add_squelch_tail=True,
        squelch_tail_sec=0.15,
        add_ctcss=True,
        ctcss_hz=151.4,
    ),
}


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tactical Radio Channel Simulator (TRCS) & Acoustic Hallucination Harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub_synth = subparsers.add_parser("synthesize", help="Generate synthetic test audio clips")
    sub_synth.add_argument("--kind", choices=["silence", "squelch", "white", "pink", "brown", "engine", "babble"], required=True)
    sub_synth.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    sub_synth.add_argument("--sample-rate", type=int, default=16000, help="Sampling frequency in Hz")
    sub_synth.add_argument("--out", type=str, required=True, help="Output WAV path")
    sub_synth.add_argument("--seed", type=int, default=42, help="Random seed")

    sub_chan = subparsers.add_parser("channel", help="Process audio file through tactical channel profile")
    sub_chan.add_argument("--in-wav", type=str, required=True, help="Input WAV file")
    sub_chan.add_argument("--profile", choices=list(TACTICAL_PROFILES.keys()), default="tactical_vhf_clear")
    sub_chan.add_argument("--codec", choices=["none", "pcm8k", "g711_mulaw", "g711_alaw", "amr_nb", "gsm_fr", "opus_nb"], default=None)
    sub_chan.add_argument("--snr", type=float, default=None, help="Override SNR in dB")
    sub_chan.add_argument("--out-wav", type=str, required=True, help="Output WAV path")

    sub_ana = subparsers.add_parser("analyze", help="Extract acoustic features and VAD decisions")
    sub_ana.add_argument("--in-wav", type=str, required=True, help="Input WAV file")
    sub_ana.add_argument("--json", action="store_true", help="Output as JSON")

    sub_grid = subparsers.add_parser("grid", help="Evaluate audio file across full telephone codec grid (E2)")
    sub_grid.add_argument("--in-wav", type=str, required=True, help="Input clean WAV file")
    sub_grid.add_argument("--json", action="store_true", help="Output as JSON")

    sub_wer = subparsers.add_parser("wer", help="Compute Word Error Rate and Entity Error Rate (E2)")
    sub_wer.add_argument("--ref", type=str, required=True, help="Reference transcript text")
    sub_wer.add_argument("--hyp", type=str, required=True, help="Hypothesis transcript text")
    sub_wer.add_argument("--callsigns", type=str, nargs="*", default=None, help="Additional scripted callsigns")
    sub_wer.add_argument("--json", action="store_true", help="Output as JSON")

    sub_e1 = subparsers.add_parser("e1", help="Run Experiment E1 Silence & Squelch Hallucination Benchmark")
    sub_e1.add_argument("--output-json", type=str, default="docs/research/e1-hallucination-ledger.json", help="Path for JSON data ledger")
    sub_e1.add_argument("--output-report", type=str, default="docs/research/E1-SILENCE-SQUELCH-HALLUCINATION-REPORT.md", help="Path for Markdown report")
    sub_e1.add_argument("--models", type=str, nargs="*", default=None, help="Models to evaluate")
    sub_e1.add_argument("--duration", type=float, default=2.5, help="Audio duration per clip")
    sub_e1.add_argument("--num-seeds", type=int, default=4, help="Seeds per condition")

    sub_e2 = subparsers.add_parser("e2", help="Run Experiment E2 Narrowband Telephone Codec Grid Benchmark")
    sub_e2.add_argument("--output-json", type=str, default="docs/research/e2-telephone-grid-ledger.json", help="Path for JSON data ledger")
    sub_e2.add_argument("--output-report", type=str, default="docs/research/E2-TELEPHONE-CODEC-GRID-REPORT.md", help="Path for Markdown report")
    sub_e2.add_argument("--models", type=str, nargs="*", default=None, help="Models to evaluate")
    sub_e2.add_argument("--batch-size", type=int, default=10, help="Batch size for inference")
    sub_e2.add_argument("--quick", action="store_true", help="Quick sanity run on 1 model and mini-grid")
    sub_e2.add_argument("--dry-run", action="store_true", help="Dry run without neural model inference (fast pipeline test)")

    args = parser.parse_args()

    if args.command == "synthesize":
        if args.kind == "silence":
            arr = generate_silence(args.duration, args.sample_rate, kind="digital")
        elif args.kind == "squelch":
            arr = generate_squelch_tail(args.duration, args.sample_rate, seed=args.seed)
        else:
            arr = generate_noise(args.duration, args.sample_rate, kind=args.kind, seed=args.seed)

        pcm16 = np.clip(arr * 32767.0, -32768.0, 32767.0).astype(np.int16)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(args.out, args.sample_rate, pcm16)
        print(f"Synthesized {args.kind} -> {args.out} ({len(arr)} samples, {args.sample_rate} Hz)")
        return 0

    elif args.command == "channel":
        cfg = TACTICAL_PROFILES[args.profile]
        if args.codec is not None:
            cfg.codec = args.codec
        if args.snr is not None:
            cfg.snr_db = args.snr

        sr, data = wavfile.read(args.in_wav)
        audio_f = ensure_mono(data)

        out_audio, audit = apply_tactical_channel(audio_f, sr, cfg)
        out_pcm16 = np.clip(out_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(args.out_wav, cfg.output_sample_rate, out_pcm16)
        print(f"Applied profile '{cfg.profile_name}' [{cfg.codec}] -> {args.out_wav}")
        return 0

    elif args.command == "analyze":
        sr, data = wavfile.read(args.in_wav)
        audio_f = ensure_mono(data)

        feats = extract_acoustic_features(audio_f, sr)
        vad = energy_vad(audio_f, sr)
        feats["vad_active_ratio"] = vad["active_ratio"]

        if args.json:
            print(json.dumps(feats, indent=2))
        else:
            for k, v in feats.items():
                print(f"{k}: {v}")
        return 0

    elif args.command == "grid":
        sr, data = wavfile.read(args.in_wav)
        audio_f = ensure_mono(data)
        results = evaluate_telephone_codec_grid(audio_f, sr)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"{'Codec':14s} | {'SNR (dB)':8s} | {'Active Ratio':12s} | {'Centroid Hz':12s} | {'Telephony Ratio':15s}")
            print("-" * 72)
            for r in results:
                c = r["codec"]
                snr_str = f"{r['target_snr_db']} dB" if r["target_snr_db"] is not None else "Clean"
                vad_r = f"{r['vad_active_ratio']:.3f}"
                cent = f"{r['acoustic_features']['spectral_centroid_hz']:.1f}"
                tel = f"{r['acoustic_features']['subband_telephony_ratio']:.3f}"
                print(f"{c:14s} | {snr_str:8s} | {vad_r:12s} | {cent:12s} | {tel:15s}")
        return 0

    elif args.command == "wer":
        wer_res = compute_wer(args.ref, args.hyp)
        eer_res = compute_entity_error_rate(args.ref, args.hyp, custom_callsigns=args.callsigns)
        merged = {**wer_res, **eer_res}
        if args.json:
            print(json.dumps(merged, indent=2))
        else:
            print(f"WER: {wer_res['wer']*100:.1f}% (Sub: {wer_res['substitutions']}, Del: {wer_res['deletions']}, Ins: {wer_res['insertions']}, Hits: {wer_res['hits']})")
            print(f"Entity Error Rate: {eer_res['entity_error_rate']*100:.1f}% (F1: {eer_res['f1_score']:.3f}, TP: {eer_res['true_positives']}, FN: {eer_res['false_negatives']}, FP: {eer_res['false_positives']})")
            print(f"Ref Entities: {eer_res['reference_entities']}")
            print(f"Hyp Entities: {eer_res['hypothesis_entities']}")
        return 0

    elif args.command == "e1":
        from scripts.research.run_e1_hallucination_benchmark import (
            DEFAULT_MODELS,
            evaluate_model_on_e1,
            render_markdown_report,
        )
        conditions = [
            "digital_silence",
            "ambient_silence",
            "squelch_burst",
            "noise_white",
            "noise_pink",
            "noise_brown",
            "noise_engine",
            "noise_babble",
            "speech_control",
        ]
        target_models = args.models or list(DEFAULT_MODELS.keys())
        seeds = [100 + i for i in range(args.num_seeds)]

        print(f"Starting TRCS E1 Hallucination Benchmark across {len(target_models)} models...")
        model_results = []
        for m_name in target_models:
            if m_name not in DEFAULT_MODELS:
                print(f"Warning: model {m_name} not in registry, skipping")
                continue
            m_path = DEFAULT_MODELS[m_name]
            res = evaluate_model_on_e1(
                model_name=m_name,
                model_path=m_path,
                conditions=conditions,
                seeds=seeds,
                duration_sec=args.duration,
            )
            model_results.append(res)

        ledger_payload = {
            "experiment_id": "E1",
            "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "protocol_reference": "docs/research/EXPERIMENT-QUEUE-2026-10.md",
            "sample_rate": 16000,
            "clip_duration_sec": args.duration,
            "conditions_evaluated": conditions,
            "seeds": seeds,
            "models": model_results,
        }

        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ledger_payload, f, indent=2)
        print(f"\n✓ Saved structured JSON ledger to: {out_json}")

        md_content = render_markdown_report(ledger_payload)
        out_report = Path(args.output_report)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        with open(out_report, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"✓ Saved markdown report to: {out_report}")
        return 0

    elif args.command == "e2":
        from scripts.research.run_e2_telephone_grid_benchmark import run_e2_pipeline
        return run_e2_pipeline(
            output_json=args.output_json,
            output_report=args.output_report,
            models=args.models,
            batch_size=args.batch_size,
            quick=args.quick,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
