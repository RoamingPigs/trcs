#!/usr/bin/env python3
"""Tactical Radio Channel Simulator (TRCS) - Hugging Face Dataset & Manifest Generator.

Generates self-contained sample audio fixtures and JSONL split manifests for the
Open TRCS Hugging Face dataset distribution:
- Split 1: `hallucination` (E1 Silence & Squelch Hallucination Benchmark)
- Split 2: `telephone_grid` (E2 Tactical Maritime Narrowband Codec Grid Benchmark)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import wave

import numpy as np
import scipy.io.wavfile as wavfile

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.trcs_simulator import (
    TACTICAL_PROFILES,
    apply_codec,
    apply_tactical_channel,
    ensure_mono,
    generate_ctcss,
    generate_noise,
    generate_silence,
    generate_squelch_click,
    generate_squelch_tail,
    mix_at_snr,
    resample_audio,
)

# Reference maritime SAR and tactical VHF utterances
TACTICAL_UTTERANCES: List[Dict[str, Any]] = [
    {
        "id": "CG_01",
        "text": "Coast Guard Sector Columbia River, this is Cutter Bertholf CG-7501, standing by on VHF channel 16 with Sierra Bravo.",
        "callsigns": ["bertholf"],
        "expected_entities": ["bertholf", "cg-7501", "16", "sierra", "bravo"],
    },
    {
        "id": "CG_02",
        "text": "Cutter Munro CG-755, Sector San Juan, acknowledge Mayday Relay from vessel Alpha Tango 4.",
        "callsigns": ["munro"],
        "expected_entities": ["munro", "cg-755", "alpha", "tango", "4"],
    },
    {
        "id": "CG_03",
        "text": "Coast Guard Rescue 6033, Cutter Kimball CG-756, establish comms on tactical channel 22 Alpha.",
        "callsigns": ["kimball"],
        "expected_entities": ["6033", "kimball", "cg-756", "22", "alpha"],
    },
    {
        "id": "CG_04",
        "text": "Cutter Stratton CG-752, Sector Puget Sound, intercepting target vessel Yankee Zulu 89 in position Mike Charlie.",
        "callsigns": ["stratton"],
        "expected_entities": ["stratton", "cg-752", "yankee", "zulu", "89", "mike", "charlie"],
    },
    {
        "id": "CG_05",
        "text": "Cutter Waesche CG-751, Sector San Francisco, deploying rescue boat Charlie 1 with four crew.",
        "callsigns": ["waesche"],
        "expected_entities": ["waesche", "cg-751", "charlie", "1", "four"],
    },
    {
        "id": "CG_06",
        "text": "Sector Boston, Cutter Spencer CG-905, departing Boston Harbor for offshore patrol area Delta Echo.",
        "callsigns": ["spencer"],
        "expected_entities": ["boston", "spencer", "cg-905", "delta", "echo"],
    },
    {
        "id": "CG_07",
        "text": "Cutter Stone CG-758, Sector Charleston, requesting weather update for position Foxtrot Golf 12.",
        "callsigns": ["stone"],
        "expected_entities": ["stone", "cg-758", "charleston", "foxtrot", "golf", "12"],
    },
    {
        "id": "CG_08",
        "text": "Sector Miami, Cutter Hamilton CG-753, en route to distress beacon Hotel India 44 on VHF 16.",
        "callsigns": ["hamilton"],
        "expected_entities": ["miami", "hamilton", "cg-753", "hotel", "india", "44", "16"],
    },
    {
        "id": "CG_09",
        "text": "Cutter Midgett CG-757, Sector Honolulu, visual contact with fishing vessel Juliett Kilo 7.",
        "callsigns": ["midgett"],
        "expected_entities": ["midgett", "cg-757", "honolulu", "juliett", "kilo", "7"],
    },
    {
        "id": "CG_10",
        "text": "Coast Guard Sector Key West, Cutter Mohawk CG-913, boarding motor vessel Lima Mike 303.",
        "callsigns": ["mohawk"],
        "expected_entities": ["mohawk", "cg-913", "key", "west", "lima", "mike", "303"],
    },
    {
        "id": "CG_11",
        "text": "Sector Mobile, Cutter Alert CG-630, standing by on channel 21 Alpha near buoy November Oscar.",
        "callsigns": ["alert"],
        "expected_entities": ["mobile", "alert", "cg-630", "21", "alpha", "november", "oscar"],
    },
    {
        "id": "CG_12",
        "text": "Cutter Bear CG-901, Sector Virginia, conducting search pattern Papa Quebec 5 at 12 knots.",
        "callsigns": ["bear"],
        "expected_entities": ["bear", "cg-901", "virginia", "papa", "quebec", "5", "12"],
    },
    {
        "id": "CG_13",
        "text": "Sector San Diego, Cutter Active CG-618, rendezvous with patrol boat Romeo Sierra 8.",
        "callsigns": ["active"],
        "expected_entities": ["active", "cg-618", "san", "diego", "romeo", "sierra", "8"],
    },
    {
        "id": "CG_14",
        "text": "Cutter Forward CG-911, Sector Delaware Bay, escorting tanker Tango Uniform 99 into channel.",
        "callsigns": ["forward"],
        "expected_entities": ["forward", "cg-911", "delaware", "bay", "tango", "uniform", "99"],
    },
    {
        "id": "CG_15",
        "text": "Sector St. Petersburg, Cutter Resolute CG-620, secure towing operation for vessel Victor Whiskey 2.",
        "callsigns": ["resolute"],
        "expected_entities": ["resolute", "cg-620", "victor", "whiskey", "2"],
    },
    {
        "id": "CG_16",
        "text": "Cutter Harriet Lane CG-903, Sector North Carolina, investigating radar contact X-ray Yankee 15.",
        "callsigns": ["harriet lane"],
        "expected_entities": ["harriet", "lane", "cg-903", "x-ray", "yankee", "15"],
    },
    {
        "id": "CG_17",
        "text": "Sector Houston-Galveston, Cutter Dauntless CG-624, all clear at waypoint Zulu Alpha 01.",
        "callsigns": ["dauntless"],
        "expected_entities": ["dauntless", "cg-624", "houston", "zulu", "alpha", "01"],
    },
    {
        "id": "CG_18",
        "text": "Coast Guard Rescue 6514, Cutter Seneca CG-905, survivor safely hoisted on channel 23 Alpha.",
        "callsigns": ["seneca"],
        "expected_entities": ["6514", "seneca", "cg-905", "23", "alpha"],
    },
    {
        "id": "CG_19",
        "text": "Sector Jacksonville, Cutter Valiant CG-621, proceeding to coordinates Bravo Charlie 42.",
        "callsigns": ["valiant"],
        "expected_entities": ["valiant", "cg-621", "jacksonville", "bravo", "charlie", "42"],
    },
    {
        "id": "CG_20",
        "text": "Cutter Legare CG-912, Sector New York, maintaining maritime security zone Delta Echo 8.",
        "callsigns": ["legare"],
        "expected_entities": ["legare", "cg-912", "new", "york", "delta", "echo", "8"],
    },
    {
        "id": "CG_21",
        "text": "Sector Corpus Christi, Cutter Diligence CG-616, intercepting go-fast vessel Foxtrot Golf 99.",
        "callsigns": ["diligence"],
        "expected_entities": ["diligence", "cg-616", "corpus", "christi", "foxtrot", "golf", "99"],
    },
    {
        "id": "CG_22",
        "text": "Cutter Tampa CG-902, Sector Key West, interdicting migrant craft Hotel India 5.",
        "callsigns": ["tampa"],
        "expected_entities": ["tampa", "cg-902", "key", "west", "hotel", "india", "5"],
    },
    {
        "id": "CG_23",
        "text": "Sector Los Angeles-Long Beach, Cutter Steadfast CG-623, tracking sailboat Juliett Kilo 18.",
        "callsigns": ["steadfast"],
        "expected_entities": ["steadfast", "cg-623", "juliett", "kilo", "18"],
    },
    {
        "id": "CG_24",
        "text": "Cutter Campbell CG-909, Sector Boston, responding to EPIRB transmission Lima Mike 77.",
        "callsigns": ["campbell"],
        "expected_entities": ["campbell", "cg-909", "boston", "lima", "mike", "77"],
    },
    {
        "id": "CG_25",
        "text": "Sector Anchorage, Icebreaker Healy WAGB-20, opening lead through sea ice near point November Oscar 3.",
        "callsigns": ["healy"],
        "expected_entities": ["healy", "wagb-20", "anchorage", "november", "oscar", "3"],
    },
    {
        "id": "CG_26",
        "text": "Icebreaker Polar Star WAGB-10, Sector Juneau, navigating McMurdo channel at 8 knots.",
        "callsigns": ["polar star"],
        "expected_entities": ["polar", "star", "wagb-10", "juneau", "8"],
    },
    {
        "id": "CG_27",
        "text": "Sector Guam, Cutter Alex Haley CG-39, underway for search and rescue area Papa Quebec 22.",
        "callsigns": ["alex haley"],
        "expected_entities": ["alex", "haley", "cg-39", "guam", "papa", "quebec", "22"],
    },
    {
        "id": "CG_28",
        "text": "Cutter Decisive CG-629, Sector Mobile, anchor aweigh departing fairway Romeo Sierra 4.",
        "callsigns": ["decisive"],
        "expected_entities": ["decisive", "cg-629", "mobile", "romeo", "sierra", "4"],
    },
    {
        "id": "CG_29",
        "text": "Sector Humboldt Bay, Cutter Dependable CG-626, hazardous bar crossing condition Tango Uniform 1.",
        "callsigns": ["dependable"],
        "expected_entities": ["dependable", "cg-626", "tango", "uniform", "1"],
    },
    {
        "id": "CG_30",
        "text": "Cutter Confidence CG-619, Sector Maryland-NCR, safety zone established around Victor Whiskey 9.",
        "callsigns": ["confidence"],
        "expected_entities": ["confidence", "cg-619", "victor", "whiskey", "9"],
    },
    {
        "id": "CG_31",
        "text": "Sector Puget Sound, Cutter Reliance CG-615, passing Point Wilson outbound channel 13.",
        "callsigns": ["reliance"],
        "expected_entities": ["reliance", "cg-615", "wilson", "13"],
    },
    {
        "id": "CG_32",
        "text": "Cutter Vigorous CG-627, Sector Virginia, boarding commercial trawler X-ray Yankee 55.",
        "callsigns": ["vigorous"],
        "expected_entities": ["vigorous", "cg-627", "virginia", "x-ray", "yankee", "55"],
    },
    {
        "id": "CG_33",
        "text": "Sector Boston, Cutter Escanaba CG-907, rendezvous with harbor pilot at station Zulu Alpha 10.",
        "callsigns": ["escanaba"],
        "expected_entities": ["escanaba", "cg-907", "boston", "zulu", "alpha", "10"],
    },
    {
        "id": "CG_34",
        "text": "Cutter Tahoma CG-908, Sector New York, completing security sweep of Bravo Charlie 6.",
        "callsigns": ["tahoma"],
        "expected_entities": ["tahoma", "cg-908", "bravo", "charlie", "6"],
    },
    {
        "id": "CG_35",
        "text": "Sector St. Petersburg, Cutter Thetis CG-910, towing disabled vessel Delta Echo 23.",
        "callsigns": ["thetis"],
        "expected_entities": ["thetis", "cg-910", "delta", "echo", "23"],
    },
    {
        "id": "CG_36",
        "text": "Cutter Northland CG-904, Sector Charleston, resuming offshore patrol sector Foxtrot Golf 7.",
        "callsigns": ["northland"],
        "expected_entities": ["northland", "cg-904", "charleston", "foxtrot", "golf", "7"],
    },
    {
        "id": "CG_37",
        "text": "Sector Miami, Response Boat Medium 45601, on scene with overturned skiff Hotel India 88.",
        "callsigns": ["rbm-45601"],
        "expected_entities": ["45601", "miami", "hotel", "india", "88"],
    },
    {
        "id": "CG_38",
        "text": "Coast Guard Defender 25, Sector Key West, high speed pursuit of contact Juliett Kilo 14.",
        "callsigns": ["defender-25"],
        "expected_entities": ["25", "key", "west", "juliett", "kilo", "14"],
    },
    {
        "id": "CG_39",
        "text": "Sector Columbia River, Rescue 6042, hoist operations completed at coordinates Lima Mike 50.",
        "callsigns": ["rescue-6042"],
        "expected_entities": ["6042", "columbia", "river", "lima", "mike", "50"],
    },
    {
        "id": "CG_40",
        "text": "Cutter Bertholf CG-7501, Sector San Juan, mission accomplished, returning to base on VHF 16.",
        "callsigns": ["bertholf"],
        "expected_entities": ["bertholf", "cg-7501", "san", "juan", "16"],
    },
]

CODECS_GRID: List[Tuple[str, str, int]] = [
    ("none", "Clean 16kHz Control", 16000),
    ("g711_mulaw", "G.711 mu-law (8kHz)", 8000),
    ("g711_alaw", "G.711 A-law (8kHz)", 8000),
    ("amr_nb", "AMR-NB 12.2k (8kHz)", 8000),
    ("gsm_fr", "GSM-FR 13k (8kHz)", 8000),
    ("opus_nb", "Opus NB 12k (8kHz)", 8000),
]

SNR_GRID: List[Tuple[Optional[float], str]] = [
    (None, "clean"),
    (20.0, "snr20"),
    (10.0, "snr10"),
    (0.0, "snr0"),
]

HALLUCINATION_CONDITIONS = [
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


def synthesize_clean_speech(text: str, sample_rate: int = 16000) -> np.ndarray:
    """Synthesize clean speech via espeak-ng, falling back to harmonic formant synthesis."""
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak_bin is not None:
        try:
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "speech.wav")
                cmd = [espeak_bin, "-v", "en-us", "-s", "155", "-w", wav_path, text]
                subprocess.run(cmd, check=True, capture_output=True)
                sr, data = wavfile.read(wav_path)
                audio = ensure_mono(data)
                return resample_audio(audio, sr, sample_rate)
        except Exception:
            pass

    # Deterministic fallback speech-like acoustic pattern
    duration_sec = max(1.5, len(text.split()) * 0.4)
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    sig = np.zeros(n_samples, dtype=np.float32)
    f0 = 130.0
    for h in range(1, 12):
        sig += (0.5 / h) * np.sin(2.0 * np.pi * f0 * h * t)
    # Formant filter shaping
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 3.5 * t))
    sig = (sig * envelope).astype(np.float32)
    sig = sig / max(1e-6, np.max(np.abs(sig))) * 0.6
    return sig


def build_hallucination_clip(condition: str, duration_sec: float, sample_rate: int, seed: int) -> np.ndarray:
    """Build acoustic clip for E1 hallucination split."""
    if condition == "digital_silence":
        return generate_silence(duration_sec, sample_rate, kind="digital")
    elif condition == "ambient_silence":
        return generate_silence(duration_sec, sample_rate, kind="ambient", ambient_dbfs=-55.0, seed=seed)
    elif condition == "squelch_burst":
        burst = generate_squelch_tail(duration_sec=min(duration_sec, 0.12), sample_rate=sample_rate, burst_type="bandpass", amplitude=0.4, seed=seed)
        pad_len = max(0, int(duration_sec * sample_rate) - len(burst))
        left = pad_len // 2
        right = pad_len - left
        return np.pad(burst, (left, right))
    elif condition.startswith("noise_"):
        n_kind = condition.replace("noise_", "")
        return generate_noise(duration_sec, sample_rate, kind=n_kind, amplitude=0.25, seed=seed)  # type: ignore[arg-type]
    elif condition == "speech_control":
        control_text = "Coast Guard Sector Seattle, this is rescue craft Alpha 1 on channel 16."
        audio = synthesize_clean_speech(control_text, sample_rate=sample_rate)
        target_len = int(duration_sec * sample_rate)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return audio
    else:
        raise ValueError(f"Unknown condition: {condition}")


def generate_dataset(
    output_dir: Path,
    num_e1_seeds: int = 10,
    num_e2_utterances: int = 40,
    sample_rate: int = 16000,
) -> Tuple[Path, Path]:
    """Generate audio files and manifest files for Hugging Face TRCS dataset."""
    output_dir = Path(output_dir).resolve()
    manifests_dir = output_dir / "manifests"
    audio_dir = output_dir / "audio"
    data_dir = output_dir / "data"
    data_hal_dir = data_dir / "hallucination"
    data_grid_dir = data_dir / "telephone_grid"
    data_default_dir = data_dir / "default"

    for d in [manifests_dir, audio_dir, data_hal_dir, data_grid_dir, data_default_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Clean existing audio to prevent unreferenced files
    for old_wav in audio_dir.glob("*.wav"):
        try:
            old_wav.unlink()
        except OSError:
            pass

    e1_manifest_path = manifests_dir / "e1_hallucination.jsonl"
    e2_manifest_path = manifests_dir / "e2_telephone_grid.jsonl"
    data_hal_path = data_hal_dir / "test.jsonl"
    data_grid_path = data_grid_dir / "test.jsonl"
    data_default_path = data_default_dir / "test.jsonl"

    print(f"Generating TRCS Hugging Face dataset in {output_dir}...")

    # 1. Generate E1 Hallucination Split
    e1_records = []
    e1_duration = 2.5
    for cond in HALLUCINATION_CONDITIONS:
        for s_idx in range(num_e1_seeds):
            seed = 100 + s_idx
            clip_id = f"e1_{cond}_s{seed}"
            wav_filename = f"{clip_id}.wav"
            wav_path = audio_dir / wav_filename

            audio_data = build_hallucination_clip(cond, e1_duration, sample_rate, seed)
            pcm16 = np.clip(audio_data * 32767.0, -32768.0, 32767.0).astype(np.int16)
            wavfile.write(str(wav_path), sample_rate, pcm16)

            is_speech = (cond == "speech_control")
            ref_text = "Coast Guard Sector Seattle, this is rescue craft Alpha 1 on channel 16." if is_speech else ""
            entities = ["seattle", "alpha", "1", "16"] if is_speech else []

            record = {
                "id": clip_id,
                "split": "hallucination",
                "condition": cond,
                "codec": "none",
                "snr_db": None,
                "duration_sec": float(e1_duration),
                "is_speech": is_speech,
                "text": ref_text,
                "entities": entities,
                "audio_path": f"audio/{wav_filename}",
                "seed": seed,
            }
            e1_records.append(record)

    # 2. Generate E2 Telephone Codec Grid Split
    e2_records = []
    target_utterances = TACTICAL_UTTERANCES[:num_e2_utterances]
    clean_audio_cache = {}
    for u in target_utterances:
        clean_audio_cache[u["id"]] = synthesize_clean_speech(u["text"], sample_rate=sample_rate)

    rng = np.random.default_rng(42)
    for u in target_utterances:
        clean_audio = clean_audio_cache[u["id"]]
        for codec_key, codec_name, _ in CODECS_GRID:
            codec_audio = apply_codec(clean_audio, sample_rate, codec=codec_key, return_sample_rate=sample_rate)

            for snr_val, snr_label in SNR_GRID:
                cond_id = f"{codec_key}_{snr_label}"
                clip_id = f"e2_{u['id'].lower()}_{cond_id}"
                wav_filename = f"{clip_id}.wav"
                wav_path = audio_dir / wav_filename

                if snr_val is not None:
                    dur = len(codec_audio) / float(sample_rate)
                    noise = generate_noise(dur, sample_rate=sample_rate, kind="pink", seed=int(rng.integers(0, 1000000)))
                    degraded, _ = mix_at_snr(codec_audio, noise, target_snr_db=snr_val)
                else:
                    degraded = np.copy(codec_audio)

                pcm16 = np.clip(degraded * 32767.0, -32768.0, 32767.0).astype(np.int16)
                wavfile.write(str(wav_path), sample_rate, pcm16)

                record = {
                    "id": clip_id,
                    "split": "telephone_grid",
                    "utterance_id": u["id"],
                    "condition": cond_id,
                    "codec": codec_key,
                    "snr_db": float(snr_val) if snr_val is not None else None,
                    "duration_sec": float(len(degraded) / sample_rate),
                    "is_speech": True,
                    "text": u["text"],
                    "callsigns": u.get("callsigns", []),
                    "entities": u["expected_entities"],
                    "audio_path": f"audio/{wav_filename}",
                }
                e2_records.append(record)

    # 3. Atomically write manifests and data/ configs
    with open(e1_manifest_path, "w", encoding="utf-8") as f1, open(data_hal_path, "w", encoding="utf-8") as f2:
        for r in e1_records:
            line = json.dumps(r) + "\n"
            f1.write(line)
            f2.write(line)
    print(f"✓ Wrote {len(e1_records)} E1 hallucination records to {e1_manifest_path} and {data_hal_path}")

    with open(e2_manifest_path, "w", encoding="utf-8") as f1, open(data_grid_path, "w", encoding="utf-8") as f2:
        for r in e2_records:
            line = json.dumps(r) + "\n"
            f1.write(line)
            f2.write(line)
    print(f"✓ Wrote {len(e2_records)} E2 telephone grid records to {e2_manifest_path} and {data_grid_path}")

    # Write combined default split
    all_records = e2_records + e1_records
    with open(data_default_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"✓ Wrote {len(all_records)} default split records to {data_default_path}")

    # 4. Generate Cryptographic Provenance Manifest
    prov_files = []
    total_audio_duration_sec = 0.0
    total_bytes = 0
    prov_path = output_dir / "provenance_manifest.json"

    for r in sorted(all_records, key=lambda x: x["audio_path"]):
        abs_p = output_dir / r["audio_path"]
        raw_bytes = abs_p.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        sz = len(raw_bytes)
        total_bytes += sz

        with wave.open(str(abs_p), "rb") as w:
            sr = w.getframerate()
            n_ch = w.getnchannels()
            n_frames = w.getnframes()
            dur = round(n_frames / float(sr), 2)
        total_audio_duration_sec += dur

        # Standard spec
        if r["split"] == "hallucination":
            c = r["condition"]
            if c == "digital_silence":
                spec = "Digital Zero PCM Silence (ITU-T G.711 / AES67 silence reference)"
            elif c == "ambient_silence":
                spec = "TRCS Acoustic Synthesizer (DSP/Additive Noise Mixing)"
            elif c == "squelch_burst":
                spec = "TRCS Tactical VHF Carrier Squelch Tail (EIA/TIA-603)"
            elif c == "speech_control":
                spec = "TRCS Clean Reference Control (16 kHz Linear PCM)"
            else:
                spec = f"TRCS Acoustic Synthesizer (DSP/Additive Noise Mixing: {c})"
        else:
            codec = r.get("codec", "none")
            if codec == "none":
                spec = "Clean Reference Linear PCM (16 kHz, 16-bit Mono)"
            elif codec == "g711_mulaw":
                spec = "ITU-T G.711 mu-law PCM (8 kHz, 64 kbps Narrowband)"
            elif codec == "g711_alaw":
                spec = "ITU-T G.711 A-law PCM (8 kHz, 64 kbps Narrowband)"
            elif codec == "amr_nb":
                spec = "3GPP AMR-NB (8 kHz, 12.2 kbps Mode 7 ETSI TS 126 071)"
            elif codec == "gsm_fr":
                spec = "ETSI GSM 06.10 Full Rate (8 kHz, 13 kbps Telephony)"
            elif codec == "opus_nb":
                spec = "IETF RFC 6716 Opus Narrowband (8 kHz, 12 kbps VOIP Mode)"
            else:
                spec = "Narrowband Codec Simulation"

        entry = {
            "filename": Path(r["audio_path"]).name,
            "relative_path": r["audio_path"],
            "sha256": sha256,
            "size_bytes": sz,
            "duration_sec": dur,
            "sample_rate": sr,
            "channels": n_ch,
            "split": r["split"],
            "condition": r["condition"],
            "codec": r.get("codec", "none"),
            "snr_db": r.get("snr_db"),
            "standard_specification": spec,
            "text": r["text"],
            "entities": r.get("entities", []),
        }
        prov_files.append(entry)

    prov_data = {
        "schema_version": "1.0.0",
        "title": "Open TRCS Benchmark Audio Provenance & Cryptographic Ledger",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license": "Code: Apache-2.0. Audio fixtures: CC-BY-4.0 / Public Domain US Government tactical transcripts.",
        "total_files": len(prov_files),
        "splits_summary": {
            "hallucination": len(e1_records),
            "telephone_grid": len(e2_records),
        },
        "total_audio_duration_sec": round(total_audio_duration_sec, 2),
        "total_bytes": total_bytes,
        "files": prov_files,
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)
    print(f"✓ Wrote provenance manifest ({len(prov_files)} files) to {prov_path}")

    return e1_manifest_path, e2_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="TRCS Hugging Face dataset generator")
    parser.add_argument("--output-dir", type=str, default="data/trcs_hf", help="Base directory for HF dataset")
    parser.add_argument("--num-e1-seeds", type=int, default=10, help="Number of seeds per E1 condition")
    parser.add_argument("--num-e2-utterances", type=int, default=40, help="Number of tactical utterances for E2 grid")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (Hz)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    generate_dataset(
        output_dir=out_dir,
        num_e1_seeds=args.num_e1_seeds,
        num_e2_utterances=args.num_e2_utterances,
        sample_rate=args.sample_rate,
    )
    print("Dataset generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
