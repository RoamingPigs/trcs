# Audio generator

Code that made the 1,050 audio files of `RoamingPigs/trcs-tactical-audio`. It was added to this repository after the full run, on 2026-10-12, so its commit is later than the analysis plan's. It is the same code, byte for byte, that produced the release: running it reproduced every SHA-256 in the dataset's `provenance_manifest.json`.

- `dataset/trcs/data_generator.py` writes the audio, the split files and the manifest.
- `scripts/research/trcs_simulator.py` holds the codec, band-pass, noise and squelch models it imports.

What each narrowband path does: resample to 8 kHz, apply a 300 to 3400 Hz Butterworth band-pass, encode and decode, return to 16 kHz. G.711 uses Python's `audioop` companding (`pip install audioop-lts` on Python 3.13 or later). GSM full-rate and Opus (12 kbit/s, voice mode) use ffmpeg with `libgsm` and `libopus`. AMR-NB uses the OpenCORE library (`libopencore-amrnb`).

**Fallbacks.** If a real codec library is missing, the simulator silently falls back to an approximation (a hand-written G.711 routine, a band-pass-only Opus path, an emulated AMR and GSM). Those outputs are not the standard codecs and do not match the release. Run `verify_regeneration.py` to find out whether any fallback ran on your machine: it compares your output to the released manifest.

## Run

```bash
cd generator
python3 dataset/trcs/data_generator.py --output-dir /tmp/trcs-regen
python3 verify_regeneration.py /tmp/trcs-regen/provenance_manifest.json <path-to-released>/provenance_manifest.json
```

Needs `numpy`, `scipy`, `espeak-ng`, `ffmpeg` (with libgsm and libopus) and `libopencore-amrnb`. The 2026-10-12 check used ffmpeg 8.0.1, Python 3.14.4 and `audioop-lts`; the version of eSpeak NG that made the original audio is not recorded. Generation takes about a minute.
