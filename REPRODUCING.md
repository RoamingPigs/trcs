# Reproducing the results

Tested with Python 3.14.4 on Linux. Expect about three hours on a 48-thread CPU and about 3 GB of downloads.

```bash
git clone https://github.com/RoamingPigs/trcs && cd trcs
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install --no-deps --target .vendor silero-vad==6.2.3

# 1. the audio, at the pinned revision, checked against its manifest
python3 -c "from trcs_eval import data; data.fetch('data/trcs', revision='8814846748e31203aa48abaf44504d5a10531dc6')"
python3 -c "from trcs_eval import data; print(len(data.verify('data/trcs')), 'mismatches')"   # must print 0

# 2. the runs (resumable; a finished model file is skipped)
python3 run_e2.py --data data/trcs --out my-results/e2
python3 run_e1.py --data data/trcs --out my-results/e1
python3 run_speech_cost.py --data data/trcs --out my-results/speech_cost

# 3. the analysis, then compare with the published one
python3 analyze.py --results my-results --out my-results/analysis.json --report my-results/REPORT.md
python3 verify_results.py --published results --mine my-results
```

Model weights come from the Hugging Face Hub at the revisions in `MODELS.md`.

## What counts as agreement

See `expected/tolerances.json`. In short: the same software on a similar machine should reproduce at least 95% of the transcripts exactly. A different CPU, GPU or library version can flip rare tokens, so the test there is whether each published headline rate lies inside your own 95% interval.

## Report what you got

Open a "Reproduction report" issue. Failed and partial reports are published the same way as successes. The rules are in `REPRODUCTION-RULES.md`.

## Not bit-stable

The audio files and the result files are bit-stable and are checked by SHA-256. Model transcripts are not guaranteed to be identical across hardware, CPU against GPU, or library versions.
